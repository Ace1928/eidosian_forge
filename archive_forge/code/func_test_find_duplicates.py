import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_find_duplicates(self):
    a = ma.array([(2, (2.0, 'B')), (1, (2.0, 'B')), (2, (2.0, 'B')), (1, (1.0, 'B')), (2, (2.0, 'B')), (2, (2.0, 'C'))], mask=[(0, (0, 0)), (0, (0, 0)), (0, (0, 0)), (0, (0, 0)), (1, (0, 0)), (0, (1, 0))], dtype=[('A', int), ('B', [('BA', float), ('BB', '|S1')])])
    test = find_duplicates(a, ignoremask=False, return_index=True)
    control = [0, 2]
    assert_equal(sorted(test[-1]), control)
    assert_equal(test[0], a[test[-1]])
    test = find_duplicates(a, key='A', return_index=True)
    control = [0, 1, 2, 3, 5]
    assert_equal(sorted(test[-1]), control)
    assert_equal(test[0], a[test[-1]])
    test = find_duplicates(a, key='B', return_index=True)
    control = [0, 1, 2, 4]
    assert_equal(sorted(test[-1]), control)
    assert_equal(test[0], a[test[-1]])
    test = find_duplicates(a, key='BA', return_index=True)
    control = [0, 1, 2, 4]
    assert_equal(sorted(test[-1]), control)
    assert_equal(test[0], a[test[-1]])
    test = find_duplicates(a, key='BB', return_index=True)
    control = [0, 1, 2, 3, 4]
    assert_equal(sorted(test[-1]), control)
    assert_equal(test[0], a[test[-1]])