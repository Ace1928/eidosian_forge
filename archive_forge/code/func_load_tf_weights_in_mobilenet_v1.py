from typing import Optional, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_mobilenet_v1 import MobileNetV1Config
def load_tf_weights_in_mobilenet_v1(model, config, tf_checkpoint_path):
    """Load TensorFlow checkpoints in a PyTorch model."""
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    init_vars = tf.train.list_variables(tf_checkpoint_path)
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f'Loading TF weight {name} with shape {shape}')
        array = tf.train.load_variable(tf_checkpoint_path, name)
        tf_weights[name] = array
    tf_to_pt_map = _build_tf_to_pytorch_map(model, config, tf_weights)
    for name, pointer in tf_to_pt_map.items():
        logger.info(f'Importing {name}')
        if name not in tf_weights:
            logger.info(f'{name} not in tf pre-trained weights, skipping')
            continue
        array = tf_weights[name]
        if 'depthwise_weights' in name:
            logger.info('Transposing depthwise')
            array = np.transpose(array, (2, 3, 0, 1))
        elif 'weights' in name:
            logger.info('Transposing')
            if len(pointer.shape) == 2:
                array = array.squeeze().transpose()
            else:
                array = np.transpose(array, (3, 2, 0, 1))
        if pointer.shape != array.shape:
            raise ValueError(f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched')
        logger.info(f'Initialize PyTorch weight {name} {array.shape}')
        pointer.data = torch.from_numpy(array)
        tf_weights.pop(name, None)
        tf_weights.pop(name + '/RMSProp', None)
        tf_weights.pop(name + '/RMSProp_1', None)
        tf_weights.pop(name + '/ExponentialMovingAverage', None)
    logger.info(f'Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}')
    return model