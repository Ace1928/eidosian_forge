import os
import re
import numpy
from .utils import ExplicitEnum, expand_dims, is_numpy_array, is_torch_tensor, logging, reshape, squeeze, tensor_size
from .utils import transpose as transpose_func
def load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path, tf_inputs=None, allow_missing_keys=False, output_loading_info=False):
    """
    Load TF 2.0 HDF5 checkpoint in a PyTorch model We use HDF5 to easily do transfer learning (see
    https://github.com/tensorflow/tensorflow/blob/ee16fcac960ae660e0e4496658a366e2f745e1f0/tensorflow/python/keras/engine/network.py#L1352-L1357).
    """
    try:
        import tensorflow as tf
        import torch
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.')
        raise
    import transformers
    from .modeling_tf_utils import load_tf_weights
    logger.info(f'Loading TensorFlow weights from {tf_checkpoint_path}')
    tf_model_class_name = 'TF' + pt_model.__class__.__name__
    tf_model_class = getattr(transformers, tf_model_class_name)
    tf_model = tf_model_class(pt_model.config)
    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs
    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)
    load_tf_weights(tf_model, tf_checkpoint_path)
    return load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info)