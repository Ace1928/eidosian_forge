import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_compare(self):
    a = np.datetime64('2000-03-12T18:00:00.000000')
    b = np.array(['2000-03-12T18:00:00.000000', '2000-03-12T17:59:59.999999', '2000-03-12T18:00:00.000001', '1970-01-11T12:00:00.909090', '2016-01-11T12:00:00.909090'], dtype='datetime64[us]')
    assert_equal(np.equal(a, b), [1, 0, 0, 0, 0])
    assert_equal(np.not_equal(a, b), [0, 1, 1, 1, 1])
    assert_equal(np.less(a, b), [0, 0, 1, 0, 1])
    assert_equal(np.less_equal(a, b), [1, 0, 1, 0, 1])
    assert_equal(np.greater(a, b), [0, 1, 0, 1, 0])
    assert_equal(np.greater_equal(a, b), [1, 1, 0, 1, 0])