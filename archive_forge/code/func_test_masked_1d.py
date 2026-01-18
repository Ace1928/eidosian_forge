import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_masked_1d(self):
    x = array(np.arange(5), mask=True)
    assert_equal(np.ma.median(x), np.ma.masked)
    assert_equal(np.ma.median(x).shape, (), 'shape mismatch')
    assert_(type(np.ma.median(x)) is np.ma.core.MaskedConstant)
    x = array(np.arange(5), mask=False)
    assert_equal(np.ma.median(x), 2.0)
    assert_equal(np.ma.median(x).shape, (), 'shape mismatch')
    assert_(type(np.ma.median(x)) is not MaskedArray)
    x = array(np.arange(5), mask=[0, 1, 0, 0, 0])
    assert_equal(np.ma.median(x), 2.5)
    assert_equal(np.ma.median(x).shape, (), 'shape mismatch')
    assert_(type(np.ma.median(x)) is not MaskedArray)
    x = array(np.arange(5), mask=[0, 1, 1, 1, 1])
    assert_equal(np.ma.median(x), 0.0)
    assert_equal(np.ma.median(x).shape, (), 'shape mismatch')
    assert_(type(np.ma.median(x)) is not MaskedArray)
    x = array(np.arange(5), mask=[0, 1, 1, 0, 0])
    assert_equal(np.ma.median(x), 3.0)
    assert_equal(np.ma.median(x).shape, (), 'shape mismatch')
    assert_(type(np.ma.median(x)) is not MaskedArray)
    x = array(np.arange(5.0), mask=[0, 1, 1, 0, 0])
    assert_equal(np.ma.median(x), 3.0)
    assert_equal(np.ma.median(x).shape, (), 'shape mismatch')
    assert_(type(np.ma.median(x)) is not MaskedArray)
    x = array(np.arange(6), mask=[0, 1, 1, 1, 1, 0])
    assert_equal(np.ma.median(x), 2.5)
    assert_equal(np.ma.median(x).shape, (), 'shape mismatch')
    assert_(type(np.ma.median(x)) is not MaskedArray)
    x = array(np.arange(6.0), mask=[0, 1, 1, 1, 1, 0])
    assert_equal(np.ma.median(x), 2.5)
    assert_equal(np.ma.median(x).shape, (), 'shape mismatch')
    assert_(type(np.ma.median(x)) is not MaskedArray)