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
@skip_if_lt_x_gpu(1)
@dist_utils.dist_init
def test_input_moved_to_cuda_device(self):
    if self.rank != 0:
        return
    dst_worker_name = dist_utils.worker_name((self.rank + 1) % self.world_size)
    t1 = torch.ones(1)
    args = (t1, 2)
    t2 = t1 * 2
    kwargs = dict(word=t2)
    for remote_module in self._create_remote_module_iter(f'{dst_worker_name}/cuda:0', modes=[ModuleCreationMode.MODULE_CTOR]):
        ret_fut = remote_module.forward_async(*args, **kwargs)
        ret = ret_fut.wait()
        self.assertEqual(ret, tuple(reversed(args + (t2,))))
        self.assertEqual(ret[0].device.type, 'cpu')
        self.assertEqual(ret[2].device.type, 'cpu')
        ret = remote_module.forward(*args, **kwargs)
        self.assertEqual(ret, tuple(reversed(args + (t2,))))
        self.assertEqual(ret[0].device.type, 'cpu')
        self.assertEqual(ret[2].device.type, 'cpu')