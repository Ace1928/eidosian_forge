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
def test_non_cont_tensors(self):
    if self.rank == 0:
        t = torch.rand(5, 5)
        t_view = t.narrow(1, 2, 2)
        self.assertFalse(t_view.is_contiguous())
        t_cont = t_view.contiguous()
        self.assertTrue(t_cont.is_contiguous())
        self.assertEqual(t_view, t_cont)
        next_rank = (self.rank + 1) % self.world_size
        t_ret = rpc.rpc_sync(worker_name(next_rank), non_cont_test, args=(t_view, t_cont))
        self.assertEqual(t_view, t_ret)
        self.assertFalse(t_ret.is_contiguous())