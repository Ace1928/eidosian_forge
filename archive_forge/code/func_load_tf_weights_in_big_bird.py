import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_big_bird import BigBirdConfig
def load_tf_weights_in_big_bird(model, tf_checkpoint_path, is_trivia_qa=False):
    """Load tf checkpoints in a pytorch model."""

    def load_tf_weights_bert(init_vars, tf_path):
        names = []
        tf_weights = {}
        for name, shape in init_vars:
            array = tf.train.load_variable(tf_path, name)
            name = name.replace('bert/encoder/LayerNorm', 'bert/embeddings/LayerNorm')
            logger.info(f'Loading TF weight {name} with shape {shape}')
            names.append(name)
            tf_weights[name] = array
        return (names, tf_weights)

    def load_tf_weights_trivia_qa(init_vars):
        names = []
        tf_weights = {}
        for i, var in enumerate(init_vars):
            name_items = var.name.split('/')
            if 'transformer_scaffold' in name_items[0]:
                layer_name_items = name_items[0].split('_')
                if len(layer_name_items) < 3:
                    layer_name_items += [0]
                name_items[0] = f'bert/encoder/layer_{layer_name_items[2]}'
            name = '/'.join([_TRIVIA_QA_MAPPING[x] if x in _TRIVIA_QA_MAPPING else x for x in name_items])[:-2]
            if 'self/attention/output' in name:
                name = name.replace('self/attention/output', 'output')
            if i >= len(init_vars) - 2:
                name = name.replace('intermediate', 'output')
            logger.info(f'Loading TF weight {name} with shape {var.shape}')
            array = var.value().numpy()
            names.append(name)
            tf_weights[name] = array
        return (names, tf_weights)
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f'Converting TensorFlow checkpoint from {tf_path}')
    init_vars = tf.saved_model.load(tf_path).variables if is_trivia_qa else tf.train.list_variables(tf_path)
    if len(init_vars) <= 0:
        raise ValueError('Loaded trained variables cannot be empty.')
    pt_names = list(model.state_dict().keys())
    if is_trivia_qa:
        names, tf_weights = load_tf_weights_trivia_qa(init_vars)
    else:
        names, tf_weights = load_tf_weights_bert(init_vars, tf_path)
    for txt_name in names:
        array = tf_weights[txt_name]
        name = txt_name.split('/')
        if any((n in ['adam_v', 'adam_m', 'AdamWeightDecayOptimizer', 'AdamWeightDecayOptimizer_1', 'global_step'] for n in name)):
            logger.info(f'Skipping {'/'.join(name)}')
            continue
        pointer = model
        pt_name = []
        for m_name in name:
            if re.fullmatch('[A-Za-z]+_\\d+', m_name):
                scope_names = re.split('_(\\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'kernel' or scope_names[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
                pt_name.append('weight')
            elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
                pointer = getattr(pointer, 'bias')
                pt_name.append('bias')
            elif scope_names[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
                pt_name.append('weight')
            elif scope_names[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
                pt_name.append('classifier')
            elif scope_names[0] == 'transform':
                pointer = getattr(pointer, 'transform')
                pt_name.append('transform')
                if 'bias' in name or 'kernel' in name:
                    pointer = getattr(pointer, 'dense')
                    pt_name.append('dense')
                elif 'beta' in name or 'gamma' in name:
                    pointer = getattr(pointer, 'LayerNorm')
                    pt_name.append('LayerNorm')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                    pt_name.append(f'{scope_names[0]}')
                except AttributeError:
                    logger.info(f'Skipping {m_name}')
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
                pt_name.append(f'{num}')
        if m_name[-11:] == '_embeddings' or m_name == 'embeddings':
            pointer = getattr(pointer, 'weight')
            pt_name.append('weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            if len(array.shape) > len(pointer.shape) and math.prod(array.shape) == math.prod(pointer.shape):
                if txt_name.endswith('attention/self/key/kernel') or txt_name.endswith('attention/self/query/kernel') or txt_name.endswith('attention/self/value/kernel'):
                    array = array.transpose(1, 0, 2).reshape(pointer.shape)
                elif txt_name.endswith('attention/output/dense/kernel'):
                    array = array.transpose(0, 2, 1).reshape(pointer.shape)
                else:
                    array = array.reshape(pointer.shape)
            if pointer.shape != array.shape:
                raise ValueError(f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched of {txt_name}.')
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pt_weight_name = '.'.join(pt_name)
        logger.info(f'Initialize PyTorch weight {pt_weight_name} from {txt_name}.')
        pointer.data = torch.from_numpy(array)
        tf_weights.pop(txt_name, None)
        pt_names.remove(pt_weight_name)
    logger.info(f'Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.')
    logger.info(f'Weights not initialized in PyTorch model: {', '.join(pt_names)}.')
    return model