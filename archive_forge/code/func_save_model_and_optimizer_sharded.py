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
def save_model_and_optimizer_sharded(model, rank, cfg, optim=None):
    """save model and optimizer via sharded_state_dict to save_dir"""
    folder_name = cfg.dist_checkpoint_root_folder + '/' + cfg.dist_checkpoint_folder + '-' + cfg.model_name
    save_dir = Path.cwd() / folder_name
    if rank == 0:
        print(f'Saving model to {save_dir}')
    distributed_writer = dist_cp.FileSystemWriter(save_dir)
    t0 = time.perf_counter()
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {'model': model.state_dict()}
        if optim is not None:
            state_dict['optim'] = FSDP.optim_state_dict(model, optim)
        dist_cp.save_state_dict(state_dict=state_dict, storage_writer=distributed_writer, planner=DefaultSavePlanner())
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f'Sharded state checkpoint saved to {save_dir}')
        print(f'Checkpoint Time = {t1 - t0:.4f}\n')