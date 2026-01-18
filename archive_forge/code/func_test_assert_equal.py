import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_assert_equal(self):
    assert_raises(AssertionError, assert_equal, np.datetime64('nat'), np.timedelta64('nat'))