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
@skip_but_pass_in_sandcastle_if(BACKEND not in DistTestCases.backend_feature['cuda'], f'The {BACKEND} backend does not support DDP communication hook on CUDA devices')
@skip_if_lt_x_gpu(int(os.environ['WORLD_SIZE']))
def test_ddp_comm_hook_logging(self):
    hooks = [default.allreduce_hook, default.fp16_compress_hook, powerSGD.powerSGD_hook, powerSGD.batched_powerSGD_hook, quantization_hooks.quantization_pertensor_hook, quantization_hooks.quantization_perchannel_hook]
    cpp_builtin_hooks = [dist.BuiltinCommHookType.ALLREDUCE, dist.BuiltinCommHookType.FP16_COMPRESS]
    for hook in hooks:
        ddp_model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1, bias=False).cuda(self.rank), device_ids=[self.rank])
        ddp_logging_data = ddp_model._get_ddp_logging_data()
        self.assertEqual(ddp_logging_data.get('comm_hook'), None)
        ddp_model.register_comm_hook(None, hook)
        ddp_logging_data = ddp_model._get_ddp_logging_data()
        self.assertEqual(ddp_logging_data.get('comm_hook'), hook.__qualname__)
    for hook in cpp_builtin_hooks:
        ddp_model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1, bias=False).cuda(self.rank), device_ids=[self.rank])
        ddp_logging_data = ddp_model._get_ddp_logging_data()
        self.assertEqual(ddp_logging_data.get('comm_hook'), None)
        ddp_model._register_builtin_comm_hook(hook)
        ddp_logging_data = ddp_model._get_ddp_logging_data()
        self.assertEqual(ddp_logging_data.get('comm_hook'), str(hook))
    ddp_model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1, bias=False).cuda(self.rank), device_ids=[self.rank])
    ddp_logging_data = ddp_model._get_ddp_logging_data()
    self.assertEqual(ddp_logging_data.get('comm_hook'), None)
    for i in range(2):
        inp = torch.ones(1, 1, device=self.rank)
        loss = ddp_model(inp).sum()
        loss.backward()
    ddp_logging_data = ddp_model._get_ddp_logging_data()
    self.assertEqual(ddp_logging_data.get('comm_hook', ''), '')