import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def nll_loss_forward(self: List[int], target: List[int], weight: Optional[List[int]], reduction: int) -> Tuple[List[int], List[int]]:
    self_dim = len(self)
    target_dim = len(target)
    assert 0 < self_dim <= 2
    assert target_dim <= 1
    no_batch_dim = self_dim == 1 and target_dim == 0
    assert no_batch_dim or self[0] == target[0]
    n_classes = self[-1]
    scalar_shape: List[int] = []
    assert weight is None or (len(weight) == 1 and weight[0] == n_classes)
    if reduction == 0 and self_dim == 2:
        reduction_shape = [self[0]]
    else:
        reduction_shape = scalar_shape
    return (reduction_shape, scalar_shape)