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
@skip_but_pass_in_sandcastle_if(BACKEND == 'nccl', 'nccl does not support DDP on CPU models')
def test_ddp_logging_data_cpu(self):

    def parse_env(var):
        return os.environ[var] if var in os.environ else 'N/A'
    dist.set_debug_level(dist.DebugLevel.INFO)
    group, group_id, rank = self._init_global_test()
    model_DDP = self._test_ddp_logging_data(is_gpu=False)
    ddp_logging_data = model_DDP._get_ddp_logging_data()
    self.assertEqual(ddp_logging_data.get('world_size'), dist.get_world_size())
    self.assertEqual(ddp_logging_data.get('rank'), dist.get_rank())
    self.assertEqual(ddp_logging_data.get('module_name'), 'Net')
    self.assertEqual(ddp_logging_data.get('device_ids'), '')
    self.assertEqual(ddp_logging_data.get('output_device'), -1)
    self.assertEqual(ddp_logging_data.get('broadcast_buffers'), 1)
    self.assertEqual(ddp_logging_data.get('bucket_cap_bytes'), 25 * 1024 * 1024)
    self.assertEqual(ddp_logging_data.get('find_unused_parameters'), 0)
    self.assertEqual(ddp_logging_data.get('gradient_as_bucket_view'), 0)
    self.assertEqual(ddp_logging_data.get('backend_name'), dist.get_backend(group_id))
    self.assertEqual(ddp_logging_data.get('iteration'), 18)
    params = list(model_DDP.parameters())
    num_params = 0
    param_size = 0
    params = list(filter(lambda parameter: parameter.requires_grad, params))
    for p in params:
        num_params += 1
        param_size += p.numel() * p.element_size()
    self.assertEqual(ddp_logging_data.get('dtypes'), 'float')
    self.assertEqual(ddp_logging_data.get('total_parameter_size_bytes'), param_size)
    self.assertEqual(ddp_logging_data.get('num_parameter_tensors'), num_params)
    self.assertEqual(ddp_logging_data.get('bucket_sizes'), str(param_size))
    self.assertEqual(ddp_logging_data.get('master_port'), parse_env('MASTER_PORT'))
    self.assertEqual(ddp_logging_data.get('master_addr'), parse_env('MASTER_ADDR'))
    self.assertEqual(ddp_logging_data.get('torch_distributed_debug'), parse_env('TORCH_DISTRIBUTED_DEBUG'))
    self.assertEqual(ddp_logging_data.get('cuda_visible_devices'), parse_env('CUDA_VISIBLE_DEVICES'))
    if ddp_logging_data.get('backend_name') == 'gloo':
        self.assertEqual(ddp_logging_data.get('gloo_socket_ifname'), parse_env('GLOO_SOCKET_IFNAME'))
        self.assertEqual(ddp_logging_data.get('gloo_device_transport'), parse_env('GLOO_DEVICE_TRANSPORT'))
        default_gloo_threads = 2
        self.assertEqual(ddp_logging_data.get('gloo_num_threads'), default_gloo_threads)
    self.assertEqual(ddp_logging_data.get('nccl_socket_ifname'), None)
    self.assertEqual(ddp_logging_data.get('nccl_blocking_wait'), None)
    self.assertEqual(ddp_logging_data.get('nccl_async_error_handling'), None)
    self.assertEqual(ddp_logging_data.get('nccl_debug'), None)
    self.assertEqual(ddp_logging_data.get('nccl_nthreads'), None)
    self.assertEqual(ddp_logging_data.get('nccl_ib_timeout'), None)
    self.assertEqual(ddp_logging_data.get('unused_parameter_size', 0), 0)
    self.assertEqual(ddp_logging_data.get('has_rebuilt_buckets'), 1)
    self.assertEqual(ddp_logging_data.get('rebuilt_bucket_sizes'), str(param_size))
    grad_ready_order = ddp_logging_data.get('prev_iteration_grad_ready_order_indices')
    expected_order = list(reversed([str(x) for x in range(3)]))
    self.assertEqual(grad_ready_order, ', '.join(expected_order))
    bucket_indices = ddp_logging_data.get('rebuilt_per_bucket_param_indices')
    self.assertEqual(bucket_indices, ' '.join(expected_order))
    self.assertGreaterEqual(ddp_logging_data.get('avg_forward_compute_time'), 1)
    self.assertGreaterEqual(ddp_logging_data.get('avg_backward_compute_time'), 1)
    self.assertGreaterEqual(ddp_logging_data.get('avg_backward_comm_time'), 1)
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
    model = LargeNet()
    model.float()
    model.fc1.double()
    model_DDP = nn.parallel.DistributedDataParallel(model, bucket_cap_mb=1.5)
    ddp_logging_data = model_DDP._get_ddp_logging_data()
    params = list(model_DDP.parameters())
    self.assertEqual(ddp_logging_data.get('bucket_cap_bytes'), int(1.5 * 1024 * 1024))
    bucket_sizes = [params[1].numel() * params[1].element_size(), params[0].numel() * params[0].element_size()]
    self.assertEqual(ddp_logging_data.get('bucket_sizes'), ', '.join((str(x) for x in bucket_sizes)))
    self.assertEqual(ddp_logging_data.get('dtypes'), 'double, float')