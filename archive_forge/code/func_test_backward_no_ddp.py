import contextlib
import enum
import logging
import os
import threading
from typing import NamedTuple
import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.nn as nn
from torch.distributed import rpc
from torch.distributed.nn import RemoteModule
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
from torch.testing._internal.dist_utils import INIT_METHOD_TEMPLATE, dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
@requires_gloo()
@dist_init
def test_backward_no_ddp(self):
    self._do_test(DdpMode.NONE)