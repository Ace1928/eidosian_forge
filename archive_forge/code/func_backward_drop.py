from builtins import isinstance
import functools
import logging
from typing import Any, List, Tuple
import torch
from torch import nn
def backward_drop(self, non_blocking: bool=True) -> None:
    with torch.cuda.stream(self._gpu_to_cpu_stream):
        self.model_shard.to(self.offload_device, non_blocking=non_blocking)