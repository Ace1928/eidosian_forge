import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def chained_operation(function):
    """
    Checks that `verify_operation` failed and if so reports a more helpful error chaining the existing
    `DistributedOperationException`.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except DistributedOperationException as e:
            operation = f'{function.__module__}.{function.__name__}'
            raise DistributedOperationException(f'Error found while calling `{operation}`. Please see the earlier error for more details.') from e
    return wrapper