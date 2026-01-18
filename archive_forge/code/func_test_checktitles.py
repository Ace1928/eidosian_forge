import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_checktitles(self):
    adtype = [(('a', 'A'), int), (('b', 'B'), bool), (('c', 'C'), float)]
    a = ma.array([(1, 2, 3)], mask=[(0, 1, 0)], dtype=adtype)
    bdtype = [(('a', 'A'), int), (('b', 'B'), bool), (('c', 'C'), float)]
    b = ma.array([(4, 5, 6)], dtype=bdtype)
    test = stack_arrays((a, b))
    control = ma.array([(1, 2, 3), (4, 5, 6)], mask=[(0, 1, 0), (0, 0, 0)], dtype=bdtype)
    assert_equal(test, control)
    assert_equal(test.mask, control.mask)