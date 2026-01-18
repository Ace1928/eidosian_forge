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
def test_get_data_parallel_params(self):
    torch.cuda.set_device(self.rank)
    model = TwoLinLayerNet().cuda()
    params_to_ignore = ['a.weight']
    torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, params_to_ignore)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
    dp_params = torch.nn.parallel.DistributedDataParallel._get_data_parallel_params(model, named_params=True)
    for name, _ in dp_params:
        self.assertNotEqual(f'module.{params_to_ignore[0]}', name)
    num_ddp_params = len(list(model.parameters())) - 1
    count = 0
    dp_params = torch.nn.parallel.DistributedDataParallel._get_data_parallel_params(model, named_params=False)
    for _ in dp_params:
        count += 1
    self.assertEqual(count, num_ddp_params)