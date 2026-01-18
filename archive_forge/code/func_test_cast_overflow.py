import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_cast_overflow(self):

    def cast():
        numpy.datetime64('1971-01-01 00:00:00.000000000000000').astype('<M8[D]')
    assert_raises(OverflowError, cast)

    def cast2():
        numpy.datetime64('2014').astype('<M8[fs]')
    assert_raises(OverflowError, cast2)