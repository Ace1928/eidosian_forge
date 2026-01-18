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
def test_local_rref_local_value(self):
    if self.rank != 0:
        return
    dst_worker_name = worker_name(self.rank)
    rref = rpc.remote(dst_worker_name, return_value, (5,), {})
    ret = rref_local_value(rref)
    self.assertEqual(ret, 5)