from typing import Dict, Tuple
import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import RRef
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@torch.jit.script
def rpc_async_call_with_timeout(dst_worker_name: str, args: Tuple[Tensor, Tensor], kwargs: Dict[str, Tensor], timeout: float):
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs, timeout)
    ret = fut.wait()
    return ret