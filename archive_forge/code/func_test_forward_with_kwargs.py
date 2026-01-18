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
@dist_utils.dist_init
def test_forward_with_kwargs(self):
    if self.rank != 0:
        return
    dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
    args = (torch.ones(1), 2)
    kwargs = dict(word='3')
    for remote_module in self._create_remote_module_iter(dst_worker_name, modes=[ModuleCreationMode.MODULE_CTOR]):
        ret_fut = remote_module.forward_async(*args, **kwargs)
        ret = ret_fut.wait()
        self.assertEqual(ret, tuple(reversed(args + ('3',))))
        ret = remote_module.forward(*args, **kwargs)
        self.assertEqual(ret, tuple(reversed(args + ('3',))))