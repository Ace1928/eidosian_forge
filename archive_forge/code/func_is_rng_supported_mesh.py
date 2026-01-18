import contextlib
import warnings
from typing import Dict, List, Optional
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._tensor.placement_types import DTensorSpec, Shard
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh
def is_rng_supported_mesh(device_mesh: DeviceMesh) -> bool:
    """Checks if the current device of `device_mesh` supports DTensor's random APIs.
    Currently DTensor Random APIs only supports cuda/cuda-like devices. We suggest
    users call this API to test the availability before using our random APIs.

    Args:
        device_mesh (:class:`DeviceMesh`): The device mesh on which we check if the
            random ops APIs are supported.

    Returns:
        A bool value. True if `device_mesh` supports DTensor Random APIs; False otherwise.

    .. warning::
        Currently we only support correct RNG on cuda/cuda-like devices.
    """
    device_handle = _get_device_handle(device_mesh.device_type)
    if device_handle and hasattr(device_handle, 'set_rng_state'):
        return True
    else:
        warnings.warn(f'DTensor random operators may not have complete support on {device_mesh.device_type} device mesh')
        return False