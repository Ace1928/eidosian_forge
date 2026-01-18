import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
@staticmethod
@check_framework_is_available
def constant_tensor(shape: List[int], value: Union[int, float]=1, dtype: Optional[Any]=None, framework: str='pt'):
    """
        Generates a constant tensor.

        Args:
            shape (`List[int]`):
                The shape of the constant tensor.
            value (`Union[int, float]`, defaults to 1):
                The value to fill the constant tensor with.
            dtype (`Optional[Any]`, defaults to `None`):
                The dtype of the constant tensor.
            framework (`str`, defaults to `"pt"`):
                The requested framework.

        Returns:
            A constant tensor in the requested framework.
        """
    if framework == 'pt':
        return torch.full(shape, value, dtype=dtype)
    elif framework == 'tf':
        return tf.constant(value, dtype=dtype, shape=shape)
    else:
        return np.full(shape, value, dtype=dtype)