import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def recv_multigpu(tensor, src_rank: int, src_gpu_index: int, group_name: str='default', n_elements: int=0):
    """Receive a tensor from a remote GPU synchronously.

    The function asssume each process owns >1 GPUs, and the sender
    process and receiver process has equal nubmer of GPUs.

    Args:
        tensor: the received tensor, located on a GPU.
        src_rank: the rank of the source process.
        src_gpu_index (int)： the index of the source gpu on the src process.
        group_name: the name of the collective group.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError('recv_multigpu call requires NCCL.')
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, src_rank)
    if src_rank == g.rank:
        raise RuntimeError("The dst_rank '{}' is self. Considering doing GPU to GPU memcpy instead?".format(src_rank))
    if n_elements < 0:
        raise RuntimeError("The n_elements '{}' should be >= 0.".format(n_elements))
    opts = types.RecvOptions()
    opts.src_rank = src_rank
    opts.src_gpu_index = src_gpu_index
    opts.n_elements = n_elements
    g.recv([tensor], opts)