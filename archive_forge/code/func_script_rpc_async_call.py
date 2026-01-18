from typing import Dict, Tuple
import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import RRef
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@torch.jit.script
def script_rpc_async_call(dst_worker_name: str, args: Tuple[Tensor, Tensor], kwargs: Dict[str, Tensor]):
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
    ret = fut.wait()
    return ret