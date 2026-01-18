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
@require_backend_is_available({'nccl'})
@skip_if_lt_x_gpu(2)
def test_nccl_backend_bool_allgather(self):
    torch.cuda.set_device(self.rank)
    inp = {0: [True, True], 1: [False, True]}
    input_tensor = torch.tensor(inp[self.rank % 2]).to(self.rank)
    input_tensor_copy = input_tensor.clone()
    tensor_list = [torch.tensor([False, False]).to(self.rank) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, input_tensor)
    self.assertEqual(len(tensor_list), dist.get_world_size())
    for i, t in enumerate(tensor_list):
        expected = torch.tensor(inp[i % 2]).to(self.rank)
        self.assertEqual(t, expected)
    self.assertEqual(input_tensor_copy, input_tensor)