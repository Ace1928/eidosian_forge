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
@dist_init(setup_rpc=False)
@skip_but_pass_in_sandcastle_if(os.environ.get('RPC_INIT_WITH_TCP', None) == '1', 'init_rpc_then_pg does not work with TCP init, see https://github.com/pytorch/pytorch/issues/41614.')
def test_init_rpc_then_pg(self):
    rpc.init_rpc(name=worker_name(self.rank), backend=self.rpc_backend, rank=self.rank, world_size=self.world_size, rpc_backend_options=self.rpc_backend_options)
    dist.init_process_group(backend='gloo', init_method=self.init_method, rank=self.rank, world_size=self.world_size)
    next_rank = (self.rank + 1) % self.world_size
    ret = rpc.rpc_sync(worker_name(next_rank), torch.add, args=(torch.ones(2, 2), 1))
    self.assertEqual(ret, torch.ones(2, 2) + 1)
    dist.barrier()
    rpc.shutdown()