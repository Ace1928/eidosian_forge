import torch.distributed as dist
from torch._C._distributed_c10d import (
from torch.futures import Future
from typing import List
from torch import Tensor
def alltoall_base(self, output_tensor: Tensor, input_tensor: Tensor, output_split_sizes: List[int], input_split_sizes: List[int], opts=AllToAllOptions()):
    return ret_work(output_tensor)