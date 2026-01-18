import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def get_data_structure(data):
    """
    Recursively gathers the information needed to rebuild a nested list/tuple/dictionary of tensors.

    Args:
        data (nested list/tuple/dictionary of `torch.Tensor`):
            The data to send to analyze.

    Returns:
        The same data structure as `data` with [`~utils.TensorInformation`] instead of tensors.
    """

    def _get_data_structure(tensor):
        return TensorInformation(shape=tensor.shape, dtype=tensor.dtype)
    return recursively_apply(_get_data_structure, data)