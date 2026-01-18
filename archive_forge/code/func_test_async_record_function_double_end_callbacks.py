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
@dist_init
def test_async_record_function_double_end_callbacks(self):
    num_sleep_seconds = 1
    if self.rank == 1:
        with _profile() as pf:
            with torch.autograd.profiler.record_function('foo') as rf:
                fut = rpc.rpc_async(worker_name(0), my_sleep_func, args=(num_sleep_seconds,))
                rf._call_end_callbacks_on_future(fut)
                with self.assertRaisesRegex(RuntimeError, 'can only be called once.'):
                    rf._call_end_callbacks_on_future(fut)
            fut.wait()