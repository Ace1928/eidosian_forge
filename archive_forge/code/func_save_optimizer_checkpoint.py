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
def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""
    print(f'--> optim state call on rank {rank}\n')
    optim_state = FSDP.full_optim_state_dict(model, optimizer)
    print(f'optim state dict ready on {rank} and len of {len(optim_state)}\n')
    if rank == 0:
        folder_name = cfg.dist_checkpoint_root_folder + '/' + cfg.dist_checkpoint_folder + '-' + cfg.model_name
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        opt_save_name = 'optimizer' + '-' + cfg.model_name + '-' + str(epoch) + '.pt'
        opt_save_full_path = save_dir / opt_save_name
        print(f'--> saving optimizer state...')
        torch.save(optim_state, opt_save_full_path)
        print(f'--> saved {opt_save_full_path} to disk')