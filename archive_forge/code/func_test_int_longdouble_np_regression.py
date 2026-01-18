import sys
from contextlib import nullcontext
import numpy as np
import pytest
from packaging.version import Version
from ..casting import (
from ..testing import suppress_warnings
def test_int_longdouble_np_regression():
    nmant = type_info(np.float64)['nmant']
    i = 2 ** (nmant + 1) - 1
    assert int(np.longdouble(i)) == i
    assert int(np.longdouble(-i)) == -i
    if nmant >= 63:
        big_int = np.uint64(2 ** 64 - 1)
        assert int(np.longdouble(big_int)) == big_int