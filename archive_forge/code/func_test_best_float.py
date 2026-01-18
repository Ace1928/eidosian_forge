import os
from platform import machine
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..casting import (
from ..testing import suppress_warnings
def test_best_float():
    """most capable type will be np.longdouble except when

    * np.longdouble has float64 precision (MSVC compiled numpy)
    * machine is sparc64 (float128 very slow)
    * np.longdouble had float64 precision when ``casting`` moduled was imported
     (precisions on windows can change, apparently)
    """
    best = best_float()
    end_of_ints = np.float64(2 ** 53)
    assert end_of_ints == end_of_ints + 1
    end_of_ints = np.longdouble(2 ** 53)
    if end_of_ints == end_of_ints + 1 or machine() == 'sparc64' or longdouble_precision_improved():
        assert best == np.float64
    else:
        assert best == np.longdouble