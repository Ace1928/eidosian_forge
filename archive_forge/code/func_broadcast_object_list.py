from collections import namedtuple
from typing import Any, Dict, List, Optional, Union
import torch
from torch.distributed import ProcessGroup
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.custom_all_reduce import custom_all_reduce
def broadcast_object_list(obj_list: List[Any], src: int=0, group: Optional[ProcessGroup]=None):
    """Broadcast the input object list."""
    group = group or torch.distributed.group.WORLD
    ranks = torch.distributed.get_process_group_ranks(group)
    assert src in ranks, f'Invalid src rank ({src})'
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return obj_list
    torch.distributed.broadcast_object_list(obj_list, src=src, group=group)
    return obj_list