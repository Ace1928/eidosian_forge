from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def none_or_in_range(a: Optional[float]) -> bool:
    return a is None or 0.0 < a <= 100.0