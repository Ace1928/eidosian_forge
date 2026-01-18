import contextlib
import warnings
from typing import Dict, List, Optional
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._tensor.placement_types import DTensorSpec, Shard
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh
def manual_seed(seed: int, device_mesh: DeviceMesh, tp_dim: int=0) -> None:
    """Sets the seed for generating random numbers for the calling rank.

    Args:
        seed (int): The desired seed.
        device_mesh (:class:`DeviceMesh`): The device mesh to set the seed.
        tp_dim (int, optional): The mesh dimension where to apply Tensor Parallel
            Default: 0

    Returns:
        None

    .. warning::
        When calling this function, :func:`manual_seed` must be called from all ranks of the
        default `ProcessGroup` even if some ranks may not be a part of the `device_mesh`,
        with the same `seed` value.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        `manual_seed` will not set its GPU device's generator seed.
        Current implementation only supports a GPU device mesh.
    """
    device_handle = _get_device_handle(device_mesh.device_type)
    if not device_handle:
        raise NotImplementedError(f'DTensor randomness only supports cuda/cuda-like device type, but got {device_mesh.device_type}')
    object_list = [seed] * dist.get_world_size()
    dist.all_gather_object(object_list, seed)
    for rank, object in enumerate(object_list):
        if seed != int(object):
            raise RuntimeError(f'calling manual_seed function over {device_mesh} but received different seed values on ranks:', f'seed on rank {dist.get_rank()} is {seed}, and seed on rank {rank} is {object}!')
    global _rng_tracker
    if not _rng_tracker:
        _rng_tracker = OffsetBasedRNGTracker(device_mesh.device_type)
    if device_mesh.get_coordinate() is not None:
        if isinstance(_rng_tracker, TensorParallelRNGTracker):
            _rng_tracker._manual_seed(device_mesh, seed, tp_dim)
        elif isinstance(_rng_tracker, OffsetBasedRNGTracker):
            _rng_tracker._manual_seed(seed)
        else:
            raise RuntimeError(f'Unknown type of cuda RNG state tracker: _rng_tracker = {_rng_tracker}')