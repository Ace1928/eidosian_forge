import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def decrement_live_tensor(self, ptr):
    self.cleanup_count -= 1
    if self.cleanup_count == 0:
        self.wait()
    else:
        self.ptr_alias_count[ptr] -= 1
        if self.ptr_alias_count[ptr] < 1 and data_ptr_to_work.get(ptr, None) == self:
            del data_ptr_to_work[ptr]