import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def initialize_tensors(data_structure):
    """
    Recursively initializes tensors from a nested list/tuple/dictionary of [`~utils.TensorInformation`].

    Returns:
        The same data structure as `data` with tensors instead of [`~utils.TensorInformation`].
    """

    def _initialize_tensor(tensor_info):
        return torch.empty(*tensor_info.shape, dtype=tensor_info.dtype)
    return recursively_apply(_initialize_tensor, data_structure, test_type=is_tensor_information)