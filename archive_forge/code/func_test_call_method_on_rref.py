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
@dist_init(setup_rpc=True)
def test_call_method_on_rref(self):
    """
        Tests that it is possible to call an instance method on a remote object
        by using rref.owner() as destination of the call.
        """
    vals = [10, 2, 5, 7]
    dst_rank = (self.rank + 1) % self.world_size
    dst_worker = worker_name(dst_rank)
    rref = rpc.remote(dst_worker, MyClass, args=(vals[0],))
    rpc.rpc_sync(rref.owner(), _call_method_on_rref, args=(MyClass.increment_value, rref, vals[1]))
    rpc.rpc_async(rref.owner(), _call_method_on_rref, args=(MyClass.increment_value, rref, vals[2])).wait()
    rpc.remote(rref.owner(), _call_method_on_rref, args=(MyClass.increment_value, rref, vals[3])).to_here()
    result = rpc.rpc_sync(dst_worker, _call_method_on_rref, args=(MyClass.get_value, rref))
    self.assertEqual(result, sum(vals))