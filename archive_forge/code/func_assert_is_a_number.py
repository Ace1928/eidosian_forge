import numbers
from typing import Any, Union
import numpy as np
def assert_is_a_number(x: Any) -> Union[int, float]:
    """Asserts that x is a number and returns it casted to an int or a float."""
    if isinstance(x, numbers.Integral):
        return int(x)
    if isinstance(x, numbers.Number):
        return float(x)
    raise TypeError('Not a number: %s' % x)