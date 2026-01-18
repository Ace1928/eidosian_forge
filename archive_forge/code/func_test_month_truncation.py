import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_month_truncation(self):
    assert_equal(np.array('1945-03-01', dtype='M8[M]'), np.array('1945-03-31', dtype='M8[M]'))
    assert_equal(np.array('1969-11-01', dtype='M8[M]'), np.array('1969-11-30T23:59:59.99999', dtype='M').astype('M8[M]'))
    assert_equal(np.array('1969-12-01', dtype='M8[M]'), np.array('1969-12-31T23:59:59.99999', dtype='M').astype('M8[M]'))
    assert_equal(np.array('1970-01-01', dtype='M8[M]'), np.array('1970-01-31T23:59:59.99999', dtype='M').astype('M8[M]'))
    assert_equal(np.array('1980-02-01', dtype='M8[M]'), np.array('1980-02-29T23:59:59.99999', dtype='M').astype('M8[M]'))