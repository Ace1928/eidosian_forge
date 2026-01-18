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
@skip_but_pass_in_sandcastle_if(BACKEND not in DistTestCases.backend_feature['subgroup'], f'The {BACKEND} backend does not support creating subgroups on CUDA devices')
@skip_if_lt_x_gpu(2)
def test_1_level_hierarchical_model_averager_equivalent_to_periodic_model_averager(self):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    rank_to_GPU = init_multigpu_helper(world_size, BACKEND)
    device_id = rank_to_GPU[rank][0]
    model = nn.Linear(1, 5, bias=False).cuda(device_id)
    param = next(model.parameters())
    tensor = torch.ones_like(param.data) * rank
    expected_avg_tensor = torch.ones_like(param.data) * sum(range(world_size)) / world_size
    period = 4
    for warmup_steps in [12, 13, 14, 15]:
        averager = hierarchicalSGD.HierarchicalModelAverager(period_group_size_dict=OrderedDict([(period, world_size)]), warmup_steps=warmup_steps)
        averager = averagers.PeriodicModelAverager(period=period, warmup_steps=warmup_steps)
        for step in range(0, 20):
            param.data = copy.deepcopy(tensor)
            for params in model.parameters():
                params.grad = torch.ones_like(param.data)
            averager.average_parameters(model.parameters())
            if step >= warmup_steps and (step - warmup_steps) % period == 0:
                self.assertEqual(param.data, expected_avg_tensor)
            else:
                self.assertEqual(param.data, tensor)