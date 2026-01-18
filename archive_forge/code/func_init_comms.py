import sys
from functools import wraps, partial
import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.testing._internal.common_distributed import (
def init_comms(self, init_rpc=True, backend='nccl'):
    if init_rpc:
        self.init_rpc()
    self.init_pg(backend=backend)