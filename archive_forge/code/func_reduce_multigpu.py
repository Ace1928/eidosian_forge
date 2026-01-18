import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def reduce_multigpu(tensor_list: list, dst_rank: int=0, dst_tensor: int=0, group_name: str='default', op=types.ReduceOp.SUM):
    """Reduce the tensor across the group to the destination rank
    and destination tensor.

    Args:
        tensor_list: the list of tensors to be reduced on this process;
            each tensor located on a GPU.
        dst_rank: the rank of the destination process.
        dst_tensor: the index of GPU at the destination.
        group_name: the collective group name to perform reduce.
        op: The reduce operation.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError('Multigpu calls requires NCCL and Cupy.')
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)
    _check_rank_valid(g, dst_rank)
    _check_root_tensor_valid(len(tensor_list), dst_tensor)
    opts = types.ReduceOptions()
    opts.reduceOp = op
    opts.root_rank = dst_rank
    opts.root_tensor = dst_tensor
    g.reduce(tensor_list, opts)