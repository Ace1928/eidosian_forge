import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_inner_join(self):
    a, b = (self.a, self.b)
    test = join_by('a', a, b, jointype='inner')
    control = np.array([(5, 55, 65, 105, 100), (6, 56, 66, 106, 101), (7, 57, 67, 107, 102), (8, 58, 68, 108, 103), (9, 59, 69, 109, 104)], dtype=[('a', int), ('b1', int), ('b2', int), ('c', int), ('d', int)])
    assert_equal(test, control)