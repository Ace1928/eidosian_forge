import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool=True):
    if dim_post_expr <= 0:
        assert wrap_scalar
        dim_post_expr = 1
    min = -dim_post_expr
    max = dim_post_expr - 1
    assert not (dim < min or dim > max)
    if dim < 0:
        dim += dim_post_expr
    return dim