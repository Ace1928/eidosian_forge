import torch.distributed as dist
from torch._C._distributed_c10d import (
from torch.futures import Future
from typing import List
from torch import Tensor
def getBackendName(self):
    return 'fake'