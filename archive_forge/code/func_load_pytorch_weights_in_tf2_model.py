import os
import re
import numpy
from .utils import ExplicitEnum, expand_dims, is_numpy_array, is_torch_tensor, logging, reshape, squeeze, tensor_size
from .utils import transpose as transpose_func
def load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=None, allow_missing_keys=False, output_loading_info=False, _prefix=None, tf_to_pt_weight_rename=None):
    """Load pytorch state_dict in a TF 2.0 model."""
    try:
        import tensorflow as tf
        import torch
    except ImportError:
        logger.error('Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.')
        raise
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}
    return load_pytorch_state_dict_in_tf2_model(tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info, _prefix=_prefix, tf_to_pt_weight_rename=tf_to_pt_weight_rename)