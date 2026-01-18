import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_drop_fields(self):
    a = np.array([(1, (2, 3.0)), (4, (5, 6.0))], dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])
    test = drop_fields(a, 'a')
    control = np.array([((2, 3.0),), ((5, 6.0),)], dtype=[('b', [('ba', float), ('bb', int)])])
    assert_equal(test, control)
    test = drop_fields(a, 'b')
    control = np.array([(1,), (4,)], dtype=[('a', int)])
    assert_equal(test, control)
    test = drop_fields(a, ['ba'])
    control = np.array([(1, (3.0,)), (4, (6.0,))], dtype=[('a', int), ('b', [('bb', int)])])
    assert_equal(test, control)
    test = drop_fields(a, ['ba', 'bb'])
    control = np.array([(1,), (4,)], dtype=[('a', int)])
    assert_equal(test, control)
    test = drop_fields(a, ['a', 'b'])
    control = np.array([(), ()], dtype=[])
    assert_equal(test, control)