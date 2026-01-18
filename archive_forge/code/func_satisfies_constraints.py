import functools
import itertools as it
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
def satisfies_constraints(self, params):
    size, _, allocation_size = self._get_size_and_steps(params)
    num_elements = prod(size)
    assert num_elements >= 0
    allocation_bytes = prod(allocation_size, base=dtype_size(self._dtype))

    def nullable_greater(left, right):
        if left is None or right is None:
            return False
        return left > right
    return not any((nullable_greater(num_elements, self._max_elements), nullable_greater(self._min_elements, num_elements), nullable_greater(allocation_bytes, self._max_allocation_bytes)))