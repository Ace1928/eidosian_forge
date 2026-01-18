from contextlib import contextmanager
from datetime import timedelta
from functools import (
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
def mock_init_dist(rank, world_size):
    assert not dist.is_initialized()
    store = dist.HashStore()
    store.add(f'{c10d.STORE_BASED_BARRIER_PREFIX}:0', world_size - 1)
    dist.init_process_group(backend='mock_process_group', rank=rank, world_size=world_size, store=store, group_name='fake', timeout=timedelta(seconds=1))