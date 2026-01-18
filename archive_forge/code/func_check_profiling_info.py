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
def check_profiling_info(self, self_worker_name, dst_worker_name, func, rpc_event, rpc_exec_mode):
    self.assertTrue(self_worker_name in rpc_event.name)
    self.assertTrue(dst_worker_name in rpc_event.name)
    if isinstance(func, torch.jit.ScriptFunction):
        self.assertTrue(torch._jit_internal._qualified_name(func) in rpc_event.name)
    else:
        self.assertTrue(func.__name__ in rpc_event.name)
    self.assertTrue(rpc_exec_mode.value in rpc_event.name)
    self.assertEqual(rpc_event.count, 1)