from pathlib import Path
from datetime import datetime
import torch
import time
from torch.distributed.fsdp import (
from torch.distributed._shard.checkpoint import (
from torch.distributed.checkpoint.default_planner import (
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist
def load_sharded_model_single_gpu(model, model_path):
    reader = FileSystemReader(model_path)
    state_dict = {'model': model.state_dict()}
    dist_cp.load_state_dict(state_dict=state_dict, storage_reader=FileSystemReader(model_path), no_dist=True)
    model.load_state_dict(state_dict['model'])
    print(f'Sharded state checkpoint loaded from {model_path}')
    return model