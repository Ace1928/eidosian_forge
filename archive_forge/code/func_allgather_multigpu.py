import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def allgather_multigpu(output_tensor_lists: list, input_tensor_list: list, group_name: str='default'):
    """Allgather tensors from each gpus of the group into lists.

    Args:
        output_tensor_lists (List[List[tensor]]): gathered results, with shape
            must be num_gpus * world_size * shape(tensor).
        input_tensor_list: (List[tensor]): a list of tensors, with shape
            num_gpus * shape(tensor).
        group_name: the name of the collective group.

    Returns:
        None
    """
    if not types.cupy_available():
        raise RuntimeError('Multigpu calls requires NCCL and Cupy.')
    _check_tensor_lists_input(output_tensor_lists)
    _check_tensor_list_input(input_tensor_list)
    g = _check_and_get_group(group_name)
    opts = types.AllGatherOptions()
    g.allgather(output_tensor_lists, input_tensor_list, opts)