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
def test_ddp_uneven_inputs(self):
    dim = 1000
    batch = 1
    large_model = nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 32, 5), nn.ReLU(), nn.Conv2d(32, 256, 5), nn.ReLU())
    small_model = nn.Linear(dim, dim, bias=False)
    bn_net = BatchNormNet()

    class UnusedParamModule(nn.Module):

        def __init__(self, unused_params_rank):
            super().__init__()
            self.t0 = Task()
            self.t1 = Task()
            self.unused_params_rank = unused_params_rank

        def task_parameters(self):
            return (self.t0.p, self.t1.p)

        def forward(self, x, rank):
            return self.t1(self.t0(x)) if rank != self.unused_params_rank else self.t1(x)
    unjoined_rank_with_unused_params_model = UnusedParamModule(1)
    joined_rank_with_unused_params_model = UnusedParamModule(0)
    rank = self.rank
    models_to_test = [DDPUnevenTestInput(name='batch_norm_net', model=bn_net, inp=torch.ones(batch, 2, device=rank), sync_interval=1), DDPUnevenTestInput(name='large_conv_model', model=large_model, inp=torch.ones(batch, batch, dim, dim, device=rank), sync_interval=1), DDPUnevenTestInput(name='small_model', model=small_model, inp=torch.ones(batch, dim, device=rank), sync_interval=1), DDPUnevenTestInput(name='unjoined_rank_with_unused_params_model', model=unjoined_rank_with_unused_params_model, inp=(torch.ones(batch, 2, device=rank), rank), sync_interval=1), DDPUnevenTestInput(name='joined_rank_with_unused_params_model', model=joined_rank_with_unused_params_model, inp=(torch.ones(batch, 2, device=rank), rank), sync_interval=1)]
    models_with_hook = [DDPUnevenTestInput(name='small_model_allreduce_hook', model=small_model, hook=default.allreduce_hook, state=None, inp=torch.ones(batch, dim, device=rank), sync_interval=1), DDPUnevenTestInput(name='small_model_power_sgd_hook', model=small_model, hook=powerSGD.powerSGD_hook, state=powerSGD.PowerSGDState(process_group=None, matrix_approximation_rank=1, start_powerSGD_iter=1, warm_start=False, use_error_feedback=False), inp=torch.ones(batch, dim, device=rank), sync_interval=1)]
    models_to_test.extend(models_with_hook)
    if HAS_TORCHVISION:
        resnet_model = torchvision.models.resnet50()
        models_to_test.append(DDPUnevenTestInput(name='resnet_model', model=resnet_model, inp=torch.ones(1, 3, 1000, 1000), sync_interval=1))
    models_with_sync = []
    for i, test_input in enumerate(models_to_test):
        models_with_sync.append(DDPUnevenTestInput(name=test_input.name, model=test_input.model, inp=test_input.inp, sync_interval=i + 2))
    throw_on_early_term_tests = []
    for test_input in models_to_test:
        throw_on_early_term_tests.append(DDPUnevenTestInput(name=test_input.name, model=test_input.model, inp=test_input.inp, sync_interval=test_input.sync_interval, throw_on_early_termination=True))
    models_to_test.extend(models_with_sync)
    models_to_test.extend(throw_on_early_term_tests)
    baseline_num_iters = [0, 5]
    iteration_offsets = [2, 3, 10]
    num_uneven_ranks = [1]
    if dist.get_world_size() > 2:
        num_uneven_ranks.append(2)
    iteration_mappings = []
    for num_early_join_ranks in num_uneven_ranks:
        for baseline_iter in baseline_num_iters:
            for offset in iteration_offsets:
                mapping = {rank: baseline_iter for rank in range(0, num_early_join_ranks)}
                if num_early_join_ranks > 1:
                    for rank in mapping.keys():
                        if rank > 0:
                            mapping[rank] += offset // 2
                mapping.update({rank: baseline_iter + offset for rank in range(num_early_join_ranks, dist.get_world_size())})
                iteration_mappings.append(mapping)
    for test_case, iteration_mapping in itertools.product(models_to_test, iteration_mappings):
        if self.rank == 0:
            print(f'Running test: {test_case.name} sync interval\n                        {test_case.sync_interval} with iteration mapping\n                        {iteration_mapping}')
        self._run_uneven_inputs_test(test_case, iteration_mapping, find_unused_params='unused_params_model' in test_case.name)