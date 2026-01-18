from __future__ import division
import numbers
from typing import Optional, Tuple
import numpy as np
def is_single_index(slc) -> bool:
    """Is the slice equivalent to a single index?
    """
    if slc.step is None:
        step = 1
    else:
        step = slc.step
    return slc.start is not None and slc.stop is not None and (slc.start + step >= slc.stop)