import copy
import itertools
import math
import os
import random
import sys
import tempfile
import time
from collections import namedtuple, OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from functools import reduce
from typing import Union, NamedTuple, Callable, Any
import unittest
import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.model_averaging.hierarchical_model_averager as hierarchicalSGD
import torch.distributed.algorithms.model_averaging.utils as model_averaging_utils
import torch.nn as nn
import torch.nn.functional as F
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.algorithms.ddp_comm_hooks import (
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.distributed.distributed_c10d import (
from torch.distributed.utils import (
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.distributed import _dump_DDP_relevant_env_vars, _MixedPrecision
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import (
import torch.distributed.optim.post_localSGD_optimizer as post_localSGD_optimizer
from torch.utils.data.distributed import DistributedSampler
@skip_if_lt_x_gpu(2)
@skip_but_pass_in_sandcastle_if(BACKEND not in DistTestCases.backend_feature['ddp'], f'The {BACKEND} backend does not support DistributedDataParallel')
def test_ddp_broadcast_buffer(self):
    rank = self.rank
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)

    class NetWithBuffers(nn.Module):

        def __init__(self):
            super().__init__()
            self.a = nn.Linear(10, 10, bias=False)
            self.b = nn.Linear(10, 1, bias=False)
            self.register_buffer('buffer', torch.randn(1, 2))

        def forward(self, x):
            return self.b(self.a(x))
    model = NetWithBuffers().cuda(rank)
    model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
    inp = torch.randn(2, 10, device=rank)
    for i in range(2):
        if rank == 0:
            model_ddp.module.buffer = model_ddp.module.buffer + 1
        loss = model_ddp(inp).sum()
        loss.backward()
        bufs = [torch.empty_like(model_ddp.module.buffer) for _ in range(dist.get_world_size())]
        dist.all_gather(bufs, model_ddp.module.buffer)
        rank_0_buf = bufs[0]
        for buf in bufs[1:]:
            self.assertEqual(rank_0_buf, buf)