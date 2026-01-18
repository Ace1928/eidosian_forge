import numbers
from typing import Any, Sequence, Union
import numpy as np
import numpy.typing as npt
def is_a_number(x: Any) -> bool:
    """Checks if x has either a number.Number or a np.double type."""
    return isinstance(x, numbers.Number) or isinstance(x, np.double) or isinstance(x, np.integer)