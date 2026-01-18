import enum
from typing import Tuple
import torch
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils as dist_utils
from torch import Tensor, nn
from torch._jit_internal import Future
from torch.distributed.nn import RemoteModule
from torch.distributed.nn.api.remote_module import _REMOTE_MODULE_PICKLED_ATTRIBUTES
from torch.distributed.nn.api.remote_module import _RemoteModule
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@torch.jit.script
def run_forward_async(scripted_remote_module: RemoteMyModuleInterface):
    ret_fut = scripted_remote_module.forward_async(torch.ones(1), 2, '3')
    ret = ret_fut.wait()
    return ret