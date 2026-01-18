from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def none_or_greater_0(a: Optional[int]) -> bool:
    return a is None or 0 < a