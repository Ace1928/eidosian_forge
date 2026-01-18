import numbers
from typing import Any, Sequence, Union
import numpy as np
import numpy.typing as npt
def is_integral(x: Any) -> bool:
    """Checks if x has either a number.Integral or a np.integer type."""
    return isinstance(x, numbers.Integral) or isinstance(x, np.integer)