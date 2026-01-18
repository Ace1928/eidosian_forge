import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_append_on_nested(self):
    w = self.data[0]
    test = append_fields(w, 'C', data=[10, 20, 30])
    control = ma.array([(1, (2, 3.0), 10), (4, (5, 6.0), 20), (-1, (-1, -1.0), 30)], mask=[(0, (0, 0), 0), (0, (0, 0), 0), (1, (1, 1), 0)], dtype=[('a', int), ('b', [('ba', float), ('bb', int)]), ('C', int)])
    assert_equal(test, control)