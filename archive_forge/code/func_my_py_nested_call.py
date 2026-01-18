import sys
import threading
import time
from enum import Enum
import random
import torch
import torch.nn as nn
from datetime import timedelta
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributed.rpc import RRef
from torch.testing._internal.common_utils import IS_MACOS, skip_but_pass_in_sandcastle_if
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
def my_py_nested_call(t1, t2, dst, world_size, hops):
    next_dst = (dst + 1) % world_size
    if hops > 0:
        return rpc.rpc_sync(worker_name(next_dst), my_py_nested_call, args=(t1, t2, next_dst, world_size, hops - 1))
    else:
        return rpc.rpc_sync(worker_name(next_dst), my_py_add, args=(t1, t2))