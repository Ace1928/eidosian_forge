import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_days_to_pydate(self):
    assert_equal(np.array('1599', dtype='M8[D]').astype('O'), datetime.date(1599, 1, 1))
    assert_equal(np.array('1600', dtype='M8[D]').astype('O'), datetime.date(1600, 1, 1))
    assert_equal(np.array('1601', dtype='M8[D]').astype('O'), datetime.date(1601, 1, 1))
    assert_equal(np.array('1900', dtype='M8[D]').astype('O'), datetime.date(1900, 1, 1))
    assert_equal(np.array('1901', dtype='M8[D]').astype('O'), datetime.date(1901, 1, 1))
    assert_equal(np.array('2000', dtype='M8[D]').astype('O'), datetime.date(2000, 1, 1))
    assert_equal(np.array('2001', dtype='M8[D]').astype('O'), datetime.date(2001, 1, 1))
    assert_equal(np.array('1600-02-29', dtype='M8[D]').astype('O'), datetime.date(1600, 2, 29))
    assert_equal(np.array('1600-03-01', dtype='M8[D]').astype('O'), datetime.date(1600, 3, 1))
    assert_equal(np.array('2001-03-22', dtype='M8[D]').astype('O'), datetime.date(2001, 3, 22))