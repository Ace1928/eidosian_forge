import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def broadcast_multigpu(tensor_list, src_rank: int=0, src_tensor: int=0, group_name: str='default'):
    """Broadcast the tensor from a source GPU to all other GPUs.

    Args:
        tensor_list: the tensors to broadcast (src) or receive (dst).
        src_rank: the rank of the source process.
        src_tensor: the index of the source GPU on the source process.
        group_name: the collective group name to perform broadcast.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError('Multigpu calls requires NCCL and Cupy.')
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, src_rank)
    _check_root_tensor_valid(len(tensor_list), src_tensor)
    opts = types.BroadcastOptions()
    opts.root_rank = src_rank
    opts.root_tensor = src_tensor
    g.broadcast(tensor_list, opts)