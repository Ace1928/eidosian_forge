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
def load_model_sharded(model, rank, cfg):
    folder_name = cfg.dist_checkpoint_root_folder + '/' + cfg.dist_checkpoint_folder + '-' + cfg.model_name
    load_dir = Path.cwd() / folder_name
    if not load_dir.exists():
        if rank == 0:
            print(f'No sharded_state_dict checkpoint directory found...skipping')
        return
    if rank == 0:
        print(f'loading model from model path: {load_dir} ')
    reader = FileSystemReader(load_dir)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {'model': model.state_dict()}
        if rank == 0:
            ck = checkpoint.keys()
            print(f' checkpoint key len = {len(ck)} and \n keys =  {ck}')
        dist_cp.load_state_dict(state_dict=checkpoint, storage_reader=reader)
        if rank == 0:
            print(f'checkpoint after load_state_dict()')
            ck = checkpoint.keys()
            print(f' checkpoint key len = {len(ck)} and \n keys =  {ck}')
        model.load_state_dict(checkpoint['model'])
    if rank == 0:
        print(f'Sharded state checkpoint loaded from {load_dir}')