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
@dist_init(clean_shutdown=False)
@skip_but_pass_in_sandcastle_if(IS_MACOS, 'Test is flaky on MacOS since libuv error handling is not as robust as TCP')
def test_backward_node_failure(self):
    rpc._set_rpc_timeout(5)
    initialize_pg(self.file_init_method, self.rank, self.world_size)
    with dist_autograd.context() as context_id:
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        res = rpc.rpc_sync(worker_name(self._next_rank()), torch.add, args=(t1, t2))
        dist.barrier()
        if self.rank % 2 == 0:
            shutdown_error_regex = self.get_shutdown_error_regex()
            for rank in range(self.world_size):
                if rank % 2 != 0:
                    wait_until_node_failure(rank, shutdown_error_regex)
            with self.assertRaisesRegex(RuntimeError, shutdown_error_regex):
                dist_autograd.backward(context_id, [res.sum()])
        else:
            pass