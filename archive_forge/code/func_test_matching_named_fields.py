import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_matching_named_fields(self):
    _, x, _, z = self.data
    zz = np.array([('a', 10.0, 100.0), ('b', 20.0, 200.0), ('c', 30.0, 300.0)], dtype=[('A', '|S3'), ('B', float), ('C', float)])
    test = stack_arrays((z, zz))
    control = ma.array([('A', 1, -1), ('B', 2, -1), ('a', 10.0, 100.0), ('b', 20.0, 200.0), ('c', 30.0, 300.0)], dtype=[('A', '|S3'), ('B', float), ('C', float)], mask=[(0, 0, 1), (0, 0, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0)])
    assert_equal(test, control)
    assert_equal(test.mask, control.mask)
    test = stack_arrays((z, zz, x))
    ndtype = [('A', '|S3'), ('B', float), ('C', float), ('f3', int)]
    control = ma.array([('A', 1, -1, -1), ('B', 2, -1, -1), ('a', 10.0, 100.0, -1), ('b', 20.0, 200.0, -1), ('c', 30.0, 300.0, -1), (-1, -1, -1, 1), (-1, -1, -1, 2)], dtype=ndtype, mask=[(0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (1, 1, 1, 0), (1, 1, 1, 0)])
    assert_equal(test, control)
    assert_equal(test.mask, control.mask)