import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from functools import partial, reduce
import torch
import torch.distributed as dist
import weakref
from torch._C._distributed_c10d import (
from torch.distributed.distributed_c10d import _CollOp, _store_based_barrier, P2POp
from torch.futures import Future
from torch.utils import _pytree as pytree
def allgather_into_tensor_coalesced(self, output_tensor_list, input_tensor_list):
    res = None
    for o_t, i_t in zip(output_tensor_list, input_tensor_list):
        res = self._allgather_base(o_t, i_t)
    return res