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
@require_backend_is_available(DistTestCases.backend_feature['gpu'])
@skip_if_lt_x_gpu(2)
def test_ddp_device_mesh_initialization(self):
    """
            Test DDP with device_mesh initialization.
            """
    world_size = int(os.environ['WORLD_SIZE'])
    from torch.distributed.device_mesh import init_device_mesh
    device_mesh = init_device_mesh('cuda', (world_size,))
    pg = _get_default_group()
    torch.cuda.set_device(self.rank)
    model = TwoLinLayerNet().cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_mesh=device_mesh)
    self.assertEqual(ddp_model.device_mesh, device_mesh)
    self.assertEqual(ddp_model.device_mesh.get_group(mesh_dim=0), pg)
    with self.assertRaisesRegex(RuntimeError, 'Cannot specify both process_group and device_mesh arguments.'):
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, process_group=pg, device_mesh=device_mesh)
    with self.assertRaisesRegex(RuntimeError, 'Only 1D device mesh is supported,'):
        device_mesh = init_device_mesh('cuda', (2, world_size // 2))
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_mesh=device_mesh)