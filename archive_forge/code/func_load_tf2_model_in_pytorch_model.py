import os
import re
import numpy
from .utils import ExplicitEnum, expand_dims, is_numpy_array, is_torch_tensor, logging, reshape, squeeze, tensor_size
from .utils import transpose as transpose_func
def load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=False, output_loading_info=False):
    """Load TF 2.0 model in a pytorch model"""
    weights = tf_model.weights
    return load_tf2_weights_in_pytorch_model(pt_model, weights, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info)