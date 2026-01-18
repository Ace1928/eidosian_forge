import os
import sys
import time
from itertools import zip_longest
import numpy as np
from numpy.testing import assert_
import pytest
from scipy.special._testutils import assert_func_equal
def inf_to_nan(func):
    """Decorate function to return nan if it returns inf"""

    def wrap(*a, **kw):
        v = func(*a, **kw)
        if not np.isfinite(v):
            return np.nan
        return v
    return wrap