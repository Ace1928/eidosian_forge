import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def gather_object(object: Any):
    """
    Recursively gather object in a nested list/tuple/dictionary of objects from all devices.

    Args:
        object (nested list/tuple/dictionary of picklable object):
            The data to gather.

    Returns:
        The same data structure as `object` with all the objects sent to every device.
    """
    if PartialState().distributed_type == DistributedType.XLA:
        raise NotImplementedError('gather objects in TPU is not supported')
    elif PartialState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
        return _gpu_gather_object(object)
    else:
        return object