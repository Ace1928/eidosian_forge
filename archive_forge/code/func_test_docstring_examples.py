import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_docstring_examples(self):
    """test the examples given in the docstring of ma.median"""
    x = array(np.arange(8), mask=[0] * 4 + [1] * 4)
    assert_equal(np.ma.median(x), 1.5)
    assert_equal(np.ma.median(x).shape, (), 'shape mismatch')
    assert_(type(np.ma.median(x)) is not MaskedArray)
    x = array(np.arange(10).reshape(2, 5), mask=[0] * 6 + [1] * 4)
    assert_equal(np.ma.median(x), 2.5)
    assert_equal(np.ma.median(x).shape, (), 'shape mismatch')
    assert_(type(np.ma.median(x)) is not MaskedArray)
    ma_x = np.ma.median(x, axis=-1, overwrite_input=True)
    assert_equal(ma_x, [2.0, 5.0])
    assert_equal(ma_x.shape, (2,), 'shape mismatch')
    assert_(type(ma_x) is MaskedArray)