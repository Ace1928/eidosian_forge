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
def test_call_python_function_remotely_from_script_not_supported(self):
    if self.rank != 0:
        return
    dst_worker_name = worker_name((self.rank + 1) % self.world_size)

    @torch.jit.script
    def rpc_async_call_remote_py_function_in_torchscript(dst_worker_name: str):
        args = ()
        kwargs = {}
        fut = rpc.rpc_async(dst_worker_name, python_function, args, kwargs)
        ret = fut.wait()
        return ret
    with self.assertRaisesRegex(RuntimeError, 'attempted to get undefined function'):
        ret = rpc_async_call_remote_py_function_in_torchscript(dst_worker_name)
        self.assertEqual(ret, 0)