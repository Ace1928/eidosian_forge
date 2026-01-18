import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def max_dim(self: List[int], dim: int, keep_dim: bool):
    out = sum_mean_dim(self, [dim], keep_dim, None)
    return (out, out)