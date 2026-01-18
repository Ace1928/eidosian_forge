import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def native_layer_norm(input: List[int], normalized_shape: List[int]) -> Tuple[List[int], List[int], List[int]]:
    reduction_shape: List[int] = []
    num_unreduced_dimensions = len(input) - len(normalized_shape)
    assert num_unreduced_dimensions >= 0
    for i in range(num_unreduced_dimensions):
        reduction_shape.append(input[i])
    for i in range(num_unreduced_dimensions, len(input)):
        reduction_shape.append(1)
    return (_copy(input), reduction_shape, reduction_shape)