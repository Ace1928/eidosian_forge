import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def slice_tensors(data, tensor_slice, process_index=None, num_processes=None):
    """
    Recursively takes a slice in a nested list/tuple/dictionary of tensors.

    Args:
        data (nested list/tuple/dictionary of `torch.Tensor`):
            The data to slice.
        tensor_slice (`slice`):
            The slice to take.

    Returns:
        The same data structure as `data` with all the tensors slices.
    """

    def _slice_tensor(tensor, tensor_slice):
        return tensor[tensor_slice]
    return recursively_apply(_slice_tensor, data, tensor_slice)