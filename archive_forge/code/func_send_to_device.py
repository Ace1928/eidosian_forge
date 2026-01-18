import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def send_to_device(tensor, device, non_blocking=False, skip_keys=None):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to send to a given device.
        device (`torch.device`):
            The device to send the data to.

    Returns:
        The same data structure as `tensor` with all tensors sent to the proper device.
    """
    if is_torch_tensor(tensor) or hasattr(tensor, 'to'):
        if device == 'npu':
            device = 'npu:0'
        if device == 'xpu':
            device = 'xpu:0'
        try:
            return tensor.to(device, non_blocking=non_blocking)
        except TypeError:
            return tensor.to(device)
        except AssertionError as error:
            if is_npu_available():
                if isinstance(device, int):
                    device = f'npu:{device}'
            else:
                raise error
        except Exception as error:
            if is_xpu_available():
                if isinstance(device, int):
                    device = f'xpu:{device}'
            else:
                raise error
        try:
            return tensor.to(device, non_blocking=non_blocking)
        except TypeError:
            return tensor.to(device)
    elif isinstance(tensor, (tuple, list)):
        return honor_type(tensor, (send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys) for t in tensor))
    elif isinstance(tensor, Mapping):
        if isinstance(skip_keys, str):
            skip_keys = [skip_keys]
        elif skip_keys is None:
            skip_keys = []
        return type(tensor)({k: t if k in skip_keys else send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys) for k, t in tensor.items()})
    else:
        return tensor