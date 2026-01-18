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
def test_ddp_logging_data_gpu(self):
    group, group_id, rank = self._init_global_test()
    model_DDP = self._test_ddp_logging_data(is_gpu=True)
    ddp_logging_data = model_DDP._get_ddp_logging_data()
    self.assertEqual(ddp_logging_data.get('device_ids'), str(rank))
    self.assertEqual(ddp_logging_data.get('output_device'), rank)
    grad_ready_order = ddp_logging_data.get('prev_iteration_grad_ready_order_indices')
    expected_order = list(reversed([str(x) for x in range(3)]))
    self.assertEqual(grad_ready_order, ', '.join(expected_order))
    bucket_indices = ddp_logging_data.get('rebuilt_per_bucket_param_indices')
    self.assertEqual(bucket_indices, ' '.join(expected_order))
    self.assertGreaterEqual(ddp_logging_data.get('avg_forward_compute_time'), 1)
    self.assertGreaterEqual(ddp_logging_data.get('avg_backward_compute_comm_overlap_time'), 1)
    self.assertGreaterEqual(ddp_logging_data.get('avg_backward_compute_time'), ddp_logging_data.get('avg_backward_compute_comm_overlap_time'))
    self.assertGreaterEqual(ddp_logging_data.get('avg_backward_comm_time'), ddp_logging_data.get('avg_backward_compute_comm_overlap_time'))
    fwd_host_side_time = ddp_logging_data.get('forward_compute_time_start')
    bwd_comp_start_host_side_time = ddp_logging_data.get('backward_compute_time_start')
    bwd_comp_end_host_side_time = ddp_logging_data.get('backward_compute_time_end')
    bwd_comm_start_host_side_time = ddp_logging_data.get('backward_comm_time_start')
    bwd_comm_end_host_side_time = ddp_logging_data.get('backward_comm_time_end')
    self.assertGreaterEqual(bwd_comm_end_host_side_time, bwd_comm_start_host_side_time)
    self.assertGreaterEqual(bwd_comm_start_host_side_time, bwd_comp_start_host_side_time)
    self.assertGreaterEqual(bwd_comp_end_host_side_time, bwd_comp_start_host_side_time)
    self.assertGreaterEqual(bwd_comp_start_host_side_time, fwd_host_side_time)