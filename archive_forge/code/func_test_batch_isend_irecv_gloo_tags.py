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
@skip_but_pass_in_sandcastle_if(BACKEND != 'gloo', 'GLOO Batch Send Recv CPU')
def test_batch_isend_irecv_gloo_tags(self):
    self._barrier()
    rank = dist.get_rank()
    p2p_op_list = []
    for src in range(0, dist.get_world_size()):
        if src == rank:
            continue
        send_tensor = _build_tensor(rank + 1)
        recv_tensor = _build_tensor(src + 1, value=-1)
        recv_op = dist.P2POp(dist.irecv, recv_tensor, src, tag=src)
        p2p_op_list.append(recv_op)
        send_op = dist.P2POp(dist.isend, send_tensor, src, tag=rank)
        p2p_op_list.append(send_op)
    reqs = dist.batch_isend_irecv(p2p_op_list)
    for req in reqs:
        req.wait()
    self._barrier()