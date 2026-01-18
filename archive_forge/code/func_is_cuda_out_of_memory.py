import gc
from typing import Any
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
def is_cuda_out_of_memory(exception: BaseException) -> bool:
    return isinstance(exception, RuntimeError) and len(exception.args) == 1 and ('CUDA' in exception.args[0]) and ('out of memory' in exception.args[0])