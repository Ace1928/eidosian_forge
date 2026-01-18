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
def load_model_checkpoint(model, rank, cfg):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""
    if rank != 0:
        return
    full_state_dict_model_path = Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    if not full_state_dict_model_path.is_file():
        print(f'model checkpoint {full_state_dict_model_path} not present. Returning...')
        return
    model_checkpoint = torch.load(full_state_dict_model_path)
    model.load_state_dict(model_checkpoint)
    print(f'model checkpoint loaded to rank0 cpu')