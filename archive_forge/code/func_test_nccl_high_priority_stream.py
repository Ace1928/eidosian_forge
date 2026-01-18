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
@skip_but_pass_in_sandcastle_if(BACKEND != 'nccl', 'Only NCCL backend supports high priority stream')
@skip_if_no_gpu
def test_nccl_high_priority_stream(self):
    group, _, rank = self._init_global_test()
    rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
    device_id = rank_to_GPU[rank][0]
    torch.cuda.set_device(device_id)
    new_port = str(MASTER_PORT + 1)
    os.environ['MASTER_PORT'] = new_port
    gen_iterator = dist.rendezvous('env://', rank, dist.get_world_size())
    store, rank, size = next(gen_iterator)
    store = dist.PrefixStore(new_port, store)
    opts = dist.ProcessGroupNCCL.Options()
    opts.is_high_priority_stream = False
    group_id = dist.ProcessGroupNCCL(store, rank, size, opts)
    self._test_broadcast_helper(group, group_id, rank, True, rank_to_GPU, True)