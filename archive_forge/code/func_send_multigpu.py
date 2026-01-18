import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def send_multigpu(tensor, dst_rank: int, dst_gpu_index: int, group_name: str='default', n_elements: int=0):
    """Send a tensor to a remote GPU synchronously.

    The function asssume each process owns >1 GPUs, and the sender
    process and receiver process has equal nubmer of GPUs.

    Args:
        tensor: the tensor to send, located on a GPU.
        dst_rank: the rank of the destination process.
        dst_gpu_index: the destination gpu index.
        group_name: the name of the collective group.
        n_elements: if specified, send the next n elements
            from the starting address of tensor.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError('send_multigpu call requires NCCL.')
    _check_single_tensor_input(tensor)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, dst_rank)
    if dst_rank == g.rank:
        raise RuntimeError("The dst_rank '{}' is self. Considering doing GPU to GPU memcpy instead?".format(dst_rank))
    if n_elements < 0:
        raise RuntimeError("The n_elements '{}' should >= 0.".format(n_elements))
    opts = types.SendOptions()
    opts.dst_rank = dst_rank
    opts.dst_gpu_index = dst_gpu_index
    opts.n_elements = n_elements
    g.send([tensor], opts)