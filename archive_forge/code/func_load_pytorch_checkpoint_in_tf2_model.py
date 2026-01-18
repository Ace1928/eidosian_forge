import os
import re
import numpy
from .utils import ExplicitEnum, expand_dims, is_numpy_array, is_torch_tensor, logging, reshape, squeeze, tensor_size
from .utils import transpose as transpose_func
def load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path, tf_inputs=None, allow_missing_keys=False, output_loading_info=False, _prefix=None, tf_to_pt_weight_rename=None):
    """Load pytorch checkpoints in a TF 2.0 model"""
    try:
        import tensorflow as tf
        import torch
        from safetensors.torch import load_file as safe_load_file
        from .pytorch_utils import is_torch_greater_or_equal_than_1_13
    except ImportError:
        logger.error('Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.')
        raise
    if isinstance(pytorch_checkpoint_path, str):
        pytorch_checkpoint_path = [pytorch_checkpoint_path]
    pt_state_dict = {}
    for path in pytorch_checkpoint_path:
        pt_path = os.path.abspath(path)
        logger.info(f'Loading PyTorch weights from {pt_path}')
        if pt_path.endswith('.safetensors'):
            state_dict = safe_load_file(pt_path)
        else:
            weights_only_kwarg = {'weights_only': True} if is_torch_greater_or_equal_than_1_13 else {}
            state_dict = torch.load(pt_path, map_location='cpu', **weights_only_kwarg)
        pt_state_dict.update(state_dict)
    logger.info(f'PyTorch checkpoint contains {sum((t.numel() for t in pt_state_dict.values())):,} parameters')
    return load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys, output_loading_info=output_loading_info, _prefix=_prefix, tf_to_pt_weight_rename=tf_to_pt_weight_rename)