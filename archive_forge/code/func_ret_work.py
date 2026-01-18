import torch.distributed as dist
from torch._C._distributed_c10d import (
from torch.futures import Future
from typing import List
from torch import Tensor
def ret_work(ret):
    fut = Future()
    fut.set_result(ret)
    return _create_work_from_future(fut)