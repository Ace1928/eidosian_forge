import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_2d_waxis(self):
    x = masked_array(np.arange(30).reshape(10, 3))
    x[:3] = x[-3:] = masked
    assert_equal(median(x), 14.5)
    assert_(type(np.ma.median(x)) is not MaskedArray)
    assert_equal(median(x, axis=0), [13.5, 14.5, 15.5])
    assert_(type(np.ma.median(x, axis=0)) is MaskedArray)
    assert_equal(median(x, axis=1), [0, 0, 0, 10, 13, 16, 19, 0, 0, 0])
    assert_(type(np.ma.median(x, axis=1)) is MaskedArray)
    assert_equal(median(x, axis=1).mask, [1, 1, 1, 0, 0, 0, 0, 1, 1, 1])