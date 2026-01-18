import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_no_r2postfix(self):
    a, b = (self.a, self.b)
    test = join_by('a', a, b, r1postfix='1', r2postfix='', jointype='inner')
    control = np.array([(0, 50, 65, 100, 100), (1, 51, 66, 101, 101), (2, 52, 67, 102, 102), (3, 53, 68, 103, 103), (4, 54, 69, 104, 104), (5, 55, 70, 105, 105), (6, 56, 71, 106, 106), (7, 57, 72, 107, 107), (8, 58, 73, 108, 108), (9, 59, 74, 109, 109)], dtype=[('a', int), ('b1', int), ('b', int), ('c', int), ('d', int)])
    assert_equal(test, control)