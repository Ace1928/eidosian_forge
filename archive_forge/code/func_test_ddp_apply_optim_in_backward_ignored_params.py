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
def test_ddp_apply_optim_in_backward_ignored_params(self):
    torch.cuda.set_device(self.rank)
    for init_before in [True, False]:
        with self.subTest(init_before=init_before):
            torch.manual_seed(self.rank)
            torch.cuda.manual_seed(self.rank)
            model = TwoLinLayerNet()
            params_to_ignore = ['a.weight']
            torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, params_to_ignore)
            if init_before:
                _apply_optimizer_in_backward(optimizer_class=torch.optim.SGD, params=model.parameters(), optimizer_kwargs={'lr': 0.03})
            net = torch.nn.parallel.DistributedDataParallel(model.cuda(self.rank), device_ids=[self.rank])
            if not init_before:
                _apply_optimizer_in_backward(optimizer_class=torch.optim.SGD, params=model.parameters(), optimizer_kwargs={'lr': 0.03})
            inp = torch.randn(1, 10)
            a, b = net(inp)
            (a.transpose(0, 1) @ b).sum().backward()
            models = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(models, model)
            rank0_model, remainder = (models[0], models[1:])
            for m in remainder:
                self.assertNotEqual(rank0_model.a.weight, m.a.weight)
                self.assertEqual(list(rank0_model.b.parameters()), list(m.b.parameters()))
                self.assertEqual(rank0_model.a.bias, m.a.bias)