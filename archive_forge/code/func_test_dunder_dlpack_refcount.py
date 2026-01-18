import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, IS_PYPY
@pytest.mark.skipif(IS_PYPY, reason="PyPy can't get refcounts.")
def test_dunder_dlpack_refcount(self):
    x = np.arange(5)
    y = x.__dlpack__()
    assert sys.getrefcount(x) == 3
    del y
    assert sys.getrefcount(x) == 2