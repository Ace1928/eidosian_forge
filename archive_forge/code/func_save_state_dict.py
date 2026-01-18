from typing import Optional
import warnings
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from .planner import SavePlanner
from .default_planner import DefaultSavePlanner
from .storage import (
from .metadata import Metadata, STATE_DICT_TYPE
from .utils import _DistWrapper
def save_state_dict(state_dict: STATE_DICT_TYPE, storage_writer: StorageWriter, process_group: Optional[dist.ProcessGroup]=None, coordinator_rank: int=0, no_dist: bool=False, planner: Optional[SavePlanner]=None) -> Metadata:
    """This method is deprecated. Please switch to 'save'."""
    warnings.warn("'save_state_dict' is deprecated and will be removed in future versions. Please use 'save' instead.")
    return _save_state_dict(state_dict, storage_writer, process_group, coordinator_rank, no_dist, planner)