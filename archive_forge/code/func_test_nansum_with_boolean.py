import os
import numpy as np
from numpy.testing import (
def test_nansum_with_boolean(self):
    a = np.zeros(2, dtype=bool)
    try:
        np.nansum(a)
    except Exception:
        raise AssertionError()