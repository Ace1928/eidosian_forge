import numbers
from typing import Any, Sequence, Union
import numpy as np
import numpy.typing as npt
def assert_is_a_number_array(x: Sequence[NumberT]) -> npt.NDArray[np.double]:
    """Asserts x is a list of numbers and converts it to np.array(np.double)."""
    result = np.empty(len(x), dtype=np.double)
    pos = 0
    for c in x:
        result[pos] = assert_is_a_number(c)
        pos += 1
    assert pos == len(x)
    return result