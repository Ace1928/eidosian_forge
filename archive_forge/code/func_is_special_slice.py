from __future__ import division
import numbers
from typing import Optional, Tuple
import numpy as np
def is_special_slice(key) -> bool:
    """Does the key contain a list, ndarray, or logical ndarray?
    """
    for elem in to_tuple(key):
        if not (isinstance(elem, (numbers.Number, slice)) or np.isscalar(elem)):
            return True
    return False