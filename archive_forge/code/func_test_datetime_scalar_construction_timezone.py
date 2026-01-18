import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_scalar_construction_timezone(self):
    with assert_warns(DeprecationWarning):
        assert_equal(np.datetime64('2000-01-01T00Z'), np.datetime64('2000-01-01T00'))
    with assert_warns(DeprecationWarning):
        assert_equal(np.datetime64('2000-01-01T00-08'), np.datetime64('2000-01-01T08'))