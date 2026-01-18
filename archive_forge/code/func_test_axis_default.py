import pytest
import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import assert_equal
from numpy.ma.core import MaskedArrayFutureWarning
import io
import textwrap
def test_axis_default(self):
    data1d = np.ma.arange(6)
    data2d = data1d.reshape(2, 3)
    ma_min = np.ma.minimum.reduce
    ma_max = np.ma.maximum.reduce
    result = assert_warns(MaskedArrayFutureWarning, ma_max, data2d)
    assert_equal(result, ma_max(data2d, axis=None))
    result = assert_warns(MaskedArrayFutureWarning, ma_min, data2d)
    assert_equal(result, ma_min(data2d, axis=None))
    result = ma_min(data1d)
    assert_equal(result, ma_min(data1d, axis=None))
    assert_equal(result, ma_min(data1d, axis=0))
    result = ma_max(data1d)
    assert_equal(result, ma_max(data1d, axis=None))
    assert_equal(result, ma_max(data1d, axis=0))