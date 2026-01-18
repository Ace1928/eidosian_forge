from typing import Dict, Tuple
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import rpc_async
from torch.testing import FileCheck
from torch.testing._internal.dist_utils import dist_init, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@torch.jit.script
def fork_add(t1, t2, dst: str):
    fut = torch.jit._fork(remote_add, t1, t2, dst)
    return torch.jit._wait(fut)