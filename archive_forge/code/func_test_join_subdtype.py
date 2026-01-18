import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_join_subdtype(self):
    foo = np.array([(1,)], dtype=[('key', int)])
    bar = np.array([(1, np.array([1, 2, 3]))], dtype=[('key', int), ('value', 'uint16', 3)])
    res = join_by('key', foo, bar)
    assert_equal(res, bar.view(ma.MaskedArray))