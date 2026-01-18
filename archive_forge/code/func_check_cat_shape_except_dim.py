import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def check_cat_shape_except_dim(first: List[int], second: List[int], dimension: int, index: int):
    first_dims = len(first)
    second_dims = len(second)
    assert first_dims == second_dims, 'Tensors must have same number of dimensions'
    for dim in range(0, first_dims):
        if dim != dimension:
            assert first[dim] == second[dim], 'Sizes of tensors must match except in dimension'