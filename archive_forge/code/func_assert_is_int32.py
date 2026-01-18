import numbers
from typing import Any, Union
import numpy as np
def assert_is_int32(x: Any) -> int:
    """Asserts that x is integer and x is in [min_int_32, max_int_32] and returns it casted to an int."""
    if not isinstance(x, numbers.Integral):
        raise TypeError('Not an integer: %s' % x)
    if x < INT32_MIN or x > INT32_MAX:
        raise OverflowError('Does not fit in an int32_t: %s' % x)
    return int(x)