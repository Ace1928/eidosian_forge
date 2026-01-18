import math
import os
import torch
import torch.distributed as dist
import bitsandbytes.functional as F
from bitsandbytes.optim.optimizer import Optimizer2State
@property
def supports_memory_efficient_fp16(self):
    return True