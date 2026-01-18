import gc
from typing import Any
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
def is_out_of_cpu_memory(exception: BaseException) -> bool:
    return isinstance(exception, RuntimeError) and len(exception.args) == 1 and ("DefaultCPUAllocator: can't allocate memory" in exception.args[0])