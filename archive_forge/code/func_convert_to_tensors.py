import json
import os
import re
import unicodedata
from json.encoder import INFINITY
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import regex
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, is_flax_available, is_tf_available, is_torch_available, logging
from ...utils.generic import _is_jax, _is_numpy
def convert_to_tensors(self, inputs, tensor_type: Optional[Union[str, TensorType]]=None, prepend_batch_axis: bool=False):
    """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                unset, no modification is done.
            prepend_batch_axis (`int`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
    if not isinstance(tensor_type, TensorType):
        tensor_type = TensorType(tensor_type)
    if tensor_type == TensorType.TENSORFLOW:
        if not is_tf_available():
            raise ImportError('Unable to convert output to TensorFlow tensors format, TensorFlow is not installed.')
        import tensorflow as tf
        as_tensor = tf.constant
        is_tensor = tf.is_tensor
    elif tensor_type == TensorType.PYTORCH:
        if not is_torch_available():
            raise ImportError('Unable to convert output to PyTorch tensors format, PyTorch is not installed.')
        import torch
        as_tensor = torch.tensor
        is_tensor = torch.is_tensor
    elif tensor_type == TensorType.JAX:
        if not is_flax_available():
            raise ImportError('Unable to convert output to JAX tensors format, JAX is not installed.')
        import jax.numpy as jnp
        as_tensor = jnp.array
        is_tensor = _is_jax
    else:
        as_tensor = np.asarray
        is_tensor = _is_numpy
    try:
        if prepend_batch_axis:
            inputs = [inputs]
        if not is_tensor(inputs):
            inputs = as_tensor(inputs)
    except:
        raise ValueError("Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length.")
    return inputs