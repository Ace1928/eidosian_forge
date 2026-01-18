import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_2d_with_missing(self):
    x = self.data
    x[-1] = masked
    x = x.reshape(3, 4)
    test = corrcoef(x)
    control = np.corrcoef(x)
    assert_almost_equal(test[:-1, :-1], control[:-1, :-1])
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, 'bias and ddof have no effect')
        assert_almost_equal(corrcoef(x, ddof=-2)[:-1, :-1], control[:-1, :-1])
        assert_almost_equal(corrcoef(x, ddof=3)[:-1, :-1], control[:-1, :-1])
        assert_almost_equal(corrcoef(x, bias=1)[:-1, :-1], control[:-1, :-1])