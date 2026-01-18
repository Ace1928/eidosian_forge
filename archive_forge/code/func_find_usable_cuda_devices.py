import os
import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, List, Optional, Union, cast
import torch
from typing_extensions import override
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.rank_zero import rank_zero_info
def find_usable_cuda_devices(num_devices: int=-1) -> List[int]:
    """Returns a list of all available and usable CUDA GPU devices.

    A GPU is considered usable if we can successfully move a tensor to the device, and this is what this function
    tests for each GPU on the system until the target number of usable devices is found.

    A subset of GPUs on the system might be used by other processes, and if the GPU is configured to operate in
    'exclusive' mode (configurable by the admin), then only one process is allowed to occupy it.

    Args:
        num_devices: The number of devices you want to request. By default, this function will return as many as there
            are usable CUDA GPU devices available.

    Warning:
        If multiple processes call this function at the same time, there can be race conditions in the case where
        both processes determine that the device is unoccupied, leading into one of them crashing later on.

    """
    if num_devices == 0:
        return []
    visible_devices = _get_all_visible_cuda_devices()
    if not visible_devices:
        raise ValueError(f'You requested to find {num_devices} devices but there are no visible CUDA devices on this machine.')
    if num_devices > len(visible_devices):
        raise ValueError(f'You requested to find {num_devices} devices but this machine only has {len(visible_devices)} GPUs.')
    available_devices = []
    unavailable_devices = []
    for gpu_idx in visible_devices:
        try:
            torch.tensor(0, device=torch.device('cuda', gpu_idx))
        except RuntimeError:
            unavailable_devices.append(gpu_idx)
            continue
        available_devices.append(gpu_idx)
        if len(available_devices) == num_devices:
            break
    if num_devices != -1 and len(available_devices) != num_devices:
        raise RuntimeError(f"You requested to find {num_devices} devices but only {len(available_devices)} are currently available. The devices {unavailable_devices} are occupied by other processes and can't be used at the moment.")
    return available_devices