import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_realm import RealmConfig
def load_tf_weights_in_realm(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f'Converting TensorFlow checkpoint from {tf_path}')
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f'Loading TF weight {name} with shape {shape}')
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        if isinstance(model, RealmReader) and 'reader' not in name:
            logger.info(f"Skipping {name} as it is not {model.__class__.__name__}'s parameter")
            continue
        if (name.startswith('bert') or name.startswith('cls')) and isinstance(model, RealmForOpenQA):
            name = name.replace('bert/', 'reader/realm/')
            name = name.replace('cls/', 'reader/cls/')
        if (name.startswith('bert') or name.startswith('cls')) and isinstance(model, RealmKnowledgeAugEncoder):
            name = name.replace('bert/', 'realm/')
        if name.startswith('reader'):
            reader_prefix = '' if isinstance(model, RealmReader) else 'reader/'
            name = name.replace('reader/module/bert/', f'{reader_prefix}realm/')
            name = name.replace('reader/module/cls/', f'{reader_prefix}cls/')
            name = name.replace('reader/dense/', f'{reader_prefix}qa_outputs/dense_intermediate/')
            name = name.replace('reader/dense_1/', f'{reader_prefix}qa_outputs/dense_output/')
            name = name.replace('reader/layer_normalization', f'{reader_prefix}qa_outputs/layer_normalization')
        if name.startswith('module/module/module/'):
            embedder_prefix = '' if isinstance(model, RealmEmbedder) else 'embedder/'
            name = name.replace('module/module/module/module/bert/', f'{embedder_prefix}realm/')
            name = name.replace('module/module/module/LayerNorm/', f'{embedder_prefix}cls/LayerNorm/')
            name = name.replace('module/module/module/dense/', f'{embedder_prefix}cls/dense/')
            name = name.replace('module/module/module/module/cls/predictions/', f'{embedder_prefix}cls/predictions/')
            name = name.replace('module/module/module/bert/', f'{embedder_prefix}realm/')
            name = name.replace('module/module/module/cls/predictions/', f'{embedder_prefix}cls/predictions/')
        elif name.startswith('module/module/'):
            embedder_prefix = '' if isinstance(model, RealmEmbedder) else 'embedder/'
            name = name.replace('module/module/LayerNorm/', f'{embedder_prefix}cls/LayerNorm/')
            name = name.replace('module/module/dense/', f'{embedder_prefix}cls/dense/')
        name = name.split('/')
        if any((n in ['adam_v', 'adam_m', 'AdamWeightDecayOptimizer', 'AdamWeightDecayOptimizer_1', 'global_step'] for n in name)):
            logger.info(f'Skipping {'/'.join(name)}')
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                scope_names = re.split('_(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'kernel' or scope_names[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f'Skipping {'/'.join(name)}')
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape, f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched'
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f'Initialize PyTorch weight {name}')
        pointer.data = torch.from_numpy(array)
    return model