import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def as_strided_writeable():
    arr = np.ones(10)
    view = as_strided(arr, writeable=False)
    assert_(not view.flags.writeable)
    view = as_strided(arr, writeable=True)
    assert_(view.flags.writeable)
    view[...] = 3
    assert_array_equal(arr, np.full_like(arr, 3))
    arr.flags.writeable = False
    view = as_strided(arr, writeable=False)
    view = as_strided(arr, writeable=True)
    assert_(not view.flags.writeable)