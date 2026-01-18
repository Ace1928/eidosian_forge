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
def test_ddp_uneven_inputs_stop_iteration_sync_bn(self):

    class ModelWithComm(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 40, bias=False)

        def forward(self, x):
            x = self.lin(x)
            dist.all_reduce(x)
            return x
    torch.cuda.set_device(self.rank)
    model_bn = BN_NET
    model_bn = nn.SyncBatchNorm.convert_sync_batchnorm(copy.deepcopy(model_bn)).cuda(self.rank)
    comm_model = ModelWithComm().cuda(self.rank)
    model_input = torch.randn(10, 2).cuda(torch.cuda.current_device())
    for model in [model_bn, comm_model]:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
        min_num_iters = 5
        if self.rank != 0:
            num_iters = min_num_iters
            exception_ctx = self.assertRaisesRegex(RuntimeError, f'Rank {self.rank} exhausted all inputs')
        else:
            num_iters = min_num_iters * 2
            exception_ctx = self.assertRaisesRegex(RuntimeError, 'Detected at least one rank that exhausted inputs.')
        n = 0
        with exception_ctx:
            with model.join(throw_on_early_termination=True):
                for i in range(num_iters):
                    loss = model(model_input).sum()
                    loss.backward()
                    self._model_step(model)
                    n += 1
        self.assertEqual(n, min_num_iters)
        self.validate_net_equivalence(model)