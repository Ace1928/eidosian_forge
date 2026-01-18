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
def test_ddp_sync_module_states(self):
    dim = 2
    rank = self.rank
    rank_to_broadcast = 1
    torch.manual_seed(rank)
    model = nn.Linear(dim, dim, bias=False)
    net = torch.nn.parallel.DistributedDataParallel(model.cuda(rank), device_ids=[self.rank], bucket_cap_mb=1)
    new_model = nn.Linear(dim, dim, bias=False).cuda(rank)
    net.module = copy.deepcopy(new_model)
    net_module_states = list(net.module.state_dict().values())
    for t in net_module_states:
        tensor_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, t)
        for i, tensor in enumerate(tensor_list):
            if i == rank:
                self.assertEqual(t, tensor)
            else:
                self.assertNotEqual(t, tensor)
    _sync_module_states(module=net.module, process_group=net.process_group, broadcast_bucket_size=net.broadcast_bucket_size, src=rank_to_broadcast, params_and_buffers_to_ignore=net.parameters_to_ignore)
    self.validate_net_equivalence(net)
    if rank == rank_to_broadcast:
        expected_states = new_model.state_dict().values()
        for t, expected in zip(net_module_states, expected_states):
            self.assertEqual(t, expected)