from typing import Callable, Optional
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from .initialize import get_model_parallel_rank, get_model_parallel_world_size
from .mappings import (
from .utils import VocabUtility, divide_and_check_no_remainder
def get_master_weight(self) -> torch.Tensor:
    return gather_from_model_parallel_region(self.weight.data)