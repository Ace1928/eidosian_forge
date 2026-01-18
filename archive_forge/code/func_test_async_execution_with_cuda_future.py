import concurrent.futures
import contextlib
import json
import os
import sys
import threading
import time
from collections import namedtuple
from functools import partial
from threading import Event
from threading import Lock
from unittest import mock
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.rpc import RRef, _get_debug_info, _rref_context_get_debug_info, WorkerInfo
from torch.distributed.rpc.api import _use_rpc_pickler, _thread_local_var, _wait_all
from torch.distributed.rpc.internal import (
from torch.futures import Future
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_utils import TemporaryFileName
from torch.autograd.profiler_legacy import profile as _profile
@skip_if_lt_x_gpu(1)
def test_async_execution_with_cuda_future(self):
    dst = worker_name((self.rank + 1) % self.world_size)
    options = self.rpc_backend_options
    options.set_device_map(dst, {'cuda:0': 'cuda:0'})
    rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=options)
    t = torch.zeros((100,), device='cuda:0')
    fut = rpc.rpc_async(dst, async_cuda_sleep_and_set_to_one, args=(t,))
    another_stream = torch.cuda.Stream('cuda:0')
    with torch.cuda.stream(another_stream):
        self.assertTrue(torch.eq(fut.wait(), 1).all().item())
    rpc.shutdown()