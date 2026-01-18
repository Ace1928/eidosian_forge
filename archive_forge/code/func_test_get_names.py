import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_get_names(self):
    ndtype = np.dtype([('A', '|S3'), ('B', float)])
    test = get_names(ndtype)
    assert_equal(test, ('A', 'B'))
    ndtype = np.dtype([('a', int), ('b', [('ba', float), ('bb', int)])])
    test = get_names(ndtype)
    assert_equal(test, ('a', ('b', ('ba', 'bb'))))
    ndtype = np.dtype([('a', int), ('b', [])])
    test = get_names(ndtype)
    assert_equal(test, ('a', ('b', ())))
    ndtype = np.dtype([])
    test = get_names(ndtype)
    assert_equal(test, ())