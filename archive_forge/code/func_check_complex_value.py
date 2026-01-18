import sys
import platform
import pytest
import numpy as np
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
def check_complex_value(f, x1, y1, x2, y2, exact=True):
    z1 = np.array([complex(x1, y1)])
    z2 = complex(x2, y2)
    with np.errstate(invalid='ignore'):
        if exact:
            assert_equal(f(z1), z2)
        else:
            assert_almost_equal(f(z1), z2)