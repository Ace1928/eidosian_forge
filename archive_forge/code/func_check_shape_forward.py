import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def check_shape_forward(input: List[int], weight_sizes: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], groups: int):
    k = len(input)
    weight_dim = len(weight_sizes)
    assert not check_non_negative(padding)
    assert not check_non_negative(stride)
    assert weight_dim == k
    assert weight_sizes[0] >= groups
    assert weight_sizes[0] % groups == 0
    assert input[1] == weight_sizes[1] * groups
    assert bias is None or (len(bias) == 1 and bias[0] == weight_sizes[0])
    for i in range(2, k):
        assert input[i] + 2 * padding[i - 2] >= dilation[i - 2] * (weight_sizes[i] - 1) + 1