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
@skip_but_pass_in_sandcastle_if(BACKEND not in DistTestCases.backend_feature['ddp'], f'The {BACKEND} backend does not support DistributedDataParallel')
@skip_if_no_gpu
def test_DistributedDataParallel(self):
    group, group_id, rank = self._init_global_test()
    rank_to_GPU = init_multigpu_helper(dist.get_world_size(), BACKEND)
    gpus = list(rank_to_GPU[rank])
    for use_bucket_view, static_graph in itertools.product((False, True), (False, True)):
        self._test_DistributedDataParallel(gpu_subset=gpus, rank=rank, gradient_as_bucket_view=use_bucket_view, static_graph=static_graph)
        self._test_DistributedDataParallel(gpu_subset=gpus, rank=rank, gradient_as_bucket_view=use_bucket_view, static_graph=static_graph, set_static_graph_twice=True)
        self._test_DistributedDataParallel(gpu_subset=gpus, rank=rank, output_device=torch.device('cuda'), gradient_as_bucket_view=use_bucket_view, static_graph=static_graph)
        gpus_list = [torch.device('cuda:' + str(i)) for i in gpus]
        self._test_DistributedDataParallel(gpu_subset=gpus_list, rank=rank, output_device=torch.device('cuda'), gradient_as_bucket_view=use_bucket_view, static_graph=static_graph)