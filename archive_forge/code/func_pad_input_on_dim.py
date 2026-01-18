import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
@classmethod
def pad_input_on_dim(cls, input_, dim: int, desired_length: Optional[int]=None, padding_length: Optional[int]=None, value: Union[int, float]=1, dtype: Optional[Any]=None):
    """
        Pads an input either to the desired length, or by a padding length.

        Args:
            input_:
                The tensor to pad.
            dim (`int`):
                The dimension along which to pad.
            desired_length (`Optional[int]`, defaults to `None`):
                The desired length along the dimension after padding.
            padding_length (`Optional[int]`, defaults to `None`):
                The length to pad along the dimension.
            value (`Union[int, float]`, defaults to 1):
                The value to use for padding.
            dtype (`Optional[Any]`, defaults to `None`):
                The dtype of the padding.

        Returns:
            The padded tensor.
        """
    if desired_length is None and padding_length is None or (desired_length is not None and padding_length is not None):
        raise ValueError('You need to provide either `desired_length` or `padding_length`')
    framework = cls._infer_framework_from_input(input_)
    shape = input_.shape
    padding_shape = list(shape)
    diff = desired_length - shape[dim] if desired_length else padding_length
    if diff <= 0:
        return input_
    padding_shape[dim] = diff
    return cls.concat_inputs([input_, cls.constant_tensor(padding_shape, value=value, dtype=dtype, framework=framework)], dim=dim)