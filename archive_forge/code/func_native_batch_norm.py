import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def native_batch_norm(input: List[int], weight: Optional[List[int]], bias: Optional[List[int]], running_mean: Optional[List[int]], running_var: Optional[List[int]], training: bool) -> Tuple[List[int], List[int], List[int]]:
    if training:
        _size = [input[1]]
    else:
        _size = [0]
    return (_copy(input), _size, _size)