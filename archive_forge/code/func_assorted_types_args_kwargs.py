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
@torch.jit.script
def assorted_types_args_kwargs(tensor_arg: Tensor, str_arg: str, int_arg: int, tensor_kwarg: Tensor=torch.tensor([2, 2]), str_kwarg: str='str_kwarg', int_kwarg: int=2):
    return (tensor_arg + tensor_kwarg, str_arg + str_kwarg, int_arg + int_kwarg)