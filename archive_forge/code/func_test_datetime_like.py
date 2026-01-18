import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_like(self):
    a = np.array([3], dtype='m8[4D]')
    b = np.array(['2012-12-21'], dtype='M8[D]')
    assert_equal(np.ones_like(a).dtype, a.dtype)
    assert_equal(np.zeros_like(a).dtype, a.dtype)
    assert_equal(np.empty_like(a).dtype, a.dtype)
    assert_equal(np.ones_like(b).dtype, b.dtype)
    assert_equal(np.zeros_like(b).dtype, b.dtype)
    assert_equal(np.empty_like(b).dtype, b.dtype)