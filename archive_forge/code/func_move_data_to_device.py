from abc import ABC
from functools import partial
from typing import Any, Callable, List, Tuple, Union
import numpy as np
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from lightning_fabric.utilities.types import _DEVICE
def move_data_to_device(batch: Any, device: _DEVICE) -> Any:
    """Transfers a collection of data to the given device. Any object that defines a method ``to(device)`` will be
    moved and all other objects in the collection will be left untouched.

    Args:
        batch: A tensor or collection of tensors or anything that has a method ``.to(...)``.
            See :func:`apply_to_collection` for a list of supported collection types.
        device: The device to which the data should be moved

    Return:
        the same collection but with all contained tensors residing on the new device.

    See Also:
        - :meth:`torch.Tensor.to`
        - :class:`torch.device`

    """
    if isinstance(device, str):
        device = torch.device(device)

    def batch_to(data: Any) -> Any:
        kwargs = {}
        if isinstance(data, Tensor) and isinstance(device, torch.device) and (device.type not in _BLOCKING_DEVICE_TYPES):
            kwargs['non_blocking'] = True
        data_output = data.to(device, **kwargs)
        if data_output is not None:
            return data_output
        return data
    return apply_to_collection(batch, dtype=_TransferableDataType, function=batch_to)