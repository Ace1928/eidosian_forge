import time
import io
from typing import Dict, List, Tuple, Any
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.autograd.profiler import record_function
from torch.distributed.rpc import RRef
from torch.distributed.rpc.internal import RPCExecMode, _build_rpc_profiling_key
from torch.futures import Future
from torch.testing._internal.common_utils import TemporaryFileName
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.autograd.profiler_legacy import profile as _profile
def test_call_fork_in_jit_with_profiling(self):
    with _profile() as prof:
        with torch.autograd.profiler.record_function('foo') as rf:
            ret = call_fork_with_profiling(rf.record)
    events = prof.function_events
    function_event = get_function_event(events, 'foo')
    self.assertEqual(function_event.name, 'foo')