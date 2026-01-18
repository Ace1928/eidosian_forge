import sys
from functools import wraps, partial
import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.testing._internal.common_distributed import (
def init_pg(self, backend='nccl'):
    if backend not in ['nccl', 'gloo', 'mpi']:
        raise RuntimeError(f'Backend {backend} not supported!')
    dist.init_process_group(backend=backend, world_size=self.world_size, rank=self.rank, init_method=f'file://{self.file_name}')
    if backend == 'nccl':
        torch.cuda.set_device(self.rank)