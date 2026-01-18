import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
@staticmethod
@check_framework_is_available
def random_mask_tensor(shape: List[int], padding_side: str='right', framework: str='pt', dtype: str='int64'):
    """
        Generates a mask tensor either right or left padded.

        Args:
            shape (`List[int]`):
                The shape of the random tensor.
            padding_side (`str`, defaults to "right"):
                The side on which the padding is applied.
            framework (`str`, defaults to `"pt"`):
                The requested framework.
            dtype (`str`, defaults to `"int64"`):
                The dtype of the generated integer tensor. Could be "int64", "int32", "int8".

        Returns:
            A random mask tensor either left padded or right padded in the requested framework.
        """
    shape = tuple(shape)
    mask_length = random.randint(1, shape[-1] - 1)
    if framework == 'pt':
        mask_tensor = torch.cat([torch.ones(*shape[:-1], shape[-1] - mask_length, dtype=DTYPE_MAPPER.pt(dtype)), torch.zeros(*shape[:-1], mask_length, dtype=DTYPE_MAPPER.pt(dtype))], dim=-1)
        if padding_side == 'left':
            mask_tensor = torch.flip(mask_tensor, [-1])
    elif framework == 'tf':
        mask_tensor = tf.concat([tf.ones((*shape[:-1], shape[-1] - mask_length), dtype=DTYPE_MAPPER.tf(dtype)), tf.zeros((*shape[:-1], mask_length), dtype=DTYPE_MAPPER.tf(dtype))], axis=-1)
        if padding_side == 'left':
            mask_tensor = tf.reverse(mask_tensor, [-1])
    else:
        mask_tensor = np.concatenate([np.ones((*shape[:-1], shape[-1] - mask_length), dtype=DTYPE_MAPPER.np(dtype)), np.zeros((*shape[:-1], mask_length), dtype=DTYPE_MAPPER.np(dtype))], axis=-1)
        if padding_side == 'left':
            mask_tensor = np.flip(mask_tensor, [-1])
    return mask_tensor