from builtins import isinstance
import functools
import logging
from typing import Any, List, Tuple
import torch
from torch import nn
def backward_load(self, non_blocking: bool=True) -> None:
    with torch.cuda.stream(self._cpu_to_gpu_stream):
        self.model_shard.to(self.device, non_blocking=non_blocking)