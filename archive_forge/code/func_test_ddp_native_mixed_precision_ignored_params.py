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
def test_ddp_native_mixed_precision_ignored_params(self):
    rank = self.rank
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    torch.cuda.set_device(rank)
    model = TwoLinLayerNet()
    model.register_buffer('buffer', torch.ones(5))
    to_ignore = ['a.weight', 'buffer']
    torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, to_ignore)
    mp_config = self._get_fp16_config()
    net = torch.nn.parallel.DistributedDataParallel(model.to(rank), device_ids=[rank], mixed_precision=mp_config, gradient_as_bucket_view=True)
    to_ignore = [f'module.{name}' for name in to_ignore]
    expected_ignored = len(to_ignore)
    n_ignored = 0
    for n, p in itertools.chain(net.named_parameters(), net.named_buffers()):
        if n in to_ignore:
            n_ignored += 1
            self.assertFalse(hasattr(p, '_mp_param'))
            self.assertFalse(hasattr(p, '_fp_param'))
        else:
            self.assertEqual(mp_config.param_dtype, p._mp_param.dtype)
            self.assertEqual(torch.float32, p._fp_param.dtype)
    self.assertEqual(expected_ignored, n_ignored)