import dataclasses
import io
import logging
import operator
from collections import ChainMap
from functools import reduce
from typing import List, Tuple, Dict, Any, Union, cast
import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.planner import (
from torch.distributed.checkpoint.metadata import (
from torch.distributed.checkpoint.planner_helpers import (
from torch.distributed.checkpoint._nested_dict import (
from torch.distributed.checkpoint._sharded_tensor_utils import (
from torch.distributed.checkpoint._dedup_tensors import dedup_tensors
from torch.distributed.checkpoint.utils import find_state_dict_object
from torch.distributed.checkpoint._traverse import set_element
def create_default_local_save_plan(state_dict: Dict[str, Any], is_coordinator: bool) -> SavePlan:
    """
    Create the ``SavePlan`` used by DefaultSavePlanner.

    On non-coordinator ranks, this function ignores tensors and non-tensor objects,
    only producing writes for ShardedTensor objects.

    On the coordinator rank, produce writes for all values.
    """
    requests = []
    for fqn, obj in state_dict.items():
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_write_items(fqn, obj)
        elif isinstance(obj, torch.Tensor) or is_coordinator:
            requests += _create_write_items(fqn, obj)
    return SavePlan(requests)