import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def index_select(self: List[int], dim: int, index: List[int]):
    dim = maybe_wrap_dim(dim, len(self))
    numel = multiply_integers(index)
    assert len(index) <= 1
    assert dim == 0 or dim < len(self)
    result_size: List[int] = []
    for i in range(len(self)):
        if dim == i:
            result_size.append(numel)
        else:
            result_size.append(self[i])
    return result_size