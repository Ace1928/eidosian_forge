import time
import io
from typing import Dict, List, Tuple, Any
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.distributed.rpc import RRef
from torch.distributed.rpc.internal import RPCExecMode, _build_rpc_profiling_key
from torch.futures import Future
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.autograd.profiler_legacy import profile as _profile
@dist_init
def test_async_function_remote_multi(self):
    dst1 = worker_name((self.rank + 1) % self.world_size)
    dst2 = worker_name((self.rank + 2) % self.world_size)
    num = 20
    rrefs = []
    for i in range(num):
        rrefs.append(rpc.remote(dst1, async_add, args=(dst2, torch.ones(2, 2), torch.ones(2, 2) * i)))
    for i in range(num):
        self.assertEqual(rrefs[i].to_here(), torch.ones(2, 2) + i)