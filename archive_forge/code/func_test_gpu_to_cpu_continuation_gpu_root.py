import sys
import threading
import time
from enum import Enum
import random
import torch
import torch.nn as nn
from datetime import timedelta
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_utils import IS_MACOS, skip_but_pass_in_sandcastle_if
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
@skip_if_lt_x_gpu(1)
@dist_init
def test_gpu_to_cpu_continuation_gpu_root(self):
    t1 = torch.rand(3, 3, requires_grad=True, device='cuda:0')
    t2 = torch.rand(3, 3, requires_grad=True)
    for i in range(3):
        t1.grad = None
        t2.grad = None
        local_grads = None
        for exec_mode in [ExecMode.LOCAL, ExecMode.RPC_SYNC]:
            with dist_autograd.context() as context_id:
                t3 = self._exec_func(exec_mode, torch.add, t2, t2)
                t4 = t3.cuda(0) + t1
                t5 = self._exec_func(exec_mode, torch.add, t4.cpu(), t2)
                t6 = t5.cuda(0) + t4
                ret = self._verify_backwards(exec_mode, [t6.sum()], context_id, local_grads, t1, t2)
                local_grads = ret if ret else local_grads