import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_flatnotmasked_contiguous(self):
    a = arange(10)
    test = flatnotmasked_contiguous(a)
    assert_equal(test, [slice(0, a.size)])
    a.mask = np.zeros(10, dtype=bool)
    assert_equal(test, [slice(0, a.size)])
    a[(a < 3) | (a > 8) | (a == 5)] = masked
    test = flatnotmasked_contiguous(a)
    assert_equal(test, [slice(3, 5), slice(6, 9)])
    a[:] = masked
    test = flatnotmasked_contiguous(a)
    assert_equal(test, [])