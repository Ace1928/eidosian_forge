import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.dist_utils import INIT_METHOD_TEMPLATE, dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import (
from torch.distributed.pipeline.sync import Pipe
@skip_if_lt_x_gpu(4)
@requires_nccl()
@dist_init
@skip_if_rocm
def test_basic_nccl_ckpt_except_last(self):
    self._run_basic_test('nccl', 'except_last', static_graph=True)