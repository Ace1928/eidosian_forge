from typing import Any
import torch
from .initialize import get_model_parallel_group
from .utils import split_tensor_along_last_dim
def scatter_to_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_)