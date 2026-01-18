import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_no_postfix(self):
    assert_raises(ValueError, join_by, 'a', self.a, self.b, r1postfix='', r2postfix='')