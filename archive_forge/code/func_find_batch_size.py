import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def find_batch_size(data):
    """
    Recursively finds the batch size in a nested list/tuple/dictionary of lists of tensors.

    Args:
        data (nested list/tuple/dictionary of `torch.Tensor`): The data from which to find the batch size.

    Returns:
        `int`: The batch size.
    """
    if isinstance(data, (tuple, list, Mapping)) and len(data) == 0:
        raise ValueError(f'Cannot find the batch size from empty {type(data)}.')
    if isinstance(data, (tuple, list)):
        return find_batch_size(data[0])
    elif isinstance(data, Mapping):
        for k in data.keys():
            return find_batch_size(data[k])
    elif not isinstance(data, torch.Tensor):
        raise TypeError(f'Can only find the batch size of tensors but got {type(data)}.')
    return data.shape[0]