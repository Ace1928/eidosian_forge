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
@skip_if_lt_x_gpu(2)
def test_cuda_future_value_on_bad_device(self):
    tensor0 = torch.zeros((100,), device='cuda:0')
    tensor1 = torch.zeros((100,), device='cuda:1')
    parent_future = Future(devices=['cuda:1'])

    def cb(fut):
        with torch.cuda.device('cuda:1'):
            torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
            tensor1.fill_(1)
            return tensor1
    child_future = parent_future.then(cb)
    with torch.cuda.device('cuda:0'):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(int(1000 * get_cycles_per_ms()))
            tensor0.fill_(1)
            parent_future.set_result(tensor0)
    with self.assertRaisesRegex(ValueError, 'The result contained tensors residing on device\\(s\\) cuda:0 which are not among the expected device\\(s\\) cuda:1'):
        parent_future.wait()
    with torch.cuda.device('cuda:1'):
        another_stream = torch.cuda.Stream()
        with torch.cuda.stream(another_stream):
            self.assertTrue(torch.eq(child_future.wait(), 1).all().item())