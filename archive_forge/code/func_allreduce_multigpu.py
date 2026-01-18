import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def allreduce_multigpu(tensor_list: list, group_name: str='default', op=types.ReduceOp.SUM):
    """Collective allreduce a list of tensors across the group.

    Args:
        tensor_list (List[tensor]): list of tensors to be allreduced,
            each on a GPU.
        group_name: the collective group name to perform allreduce.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError('Multigpu calls requires NCCL and Cupy.')
    _check_tensor_list_input(tensor_list)
    g = _check_and_get_group(group_name)
    opts = types.AllReduceOptions
    opts.reduceOp = op
    g.allreduce(tensor_list, opts)