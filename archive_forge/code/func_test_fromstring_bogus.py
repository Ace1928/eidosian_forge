import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_fromstring_bogus():
    with assert_warns(DeprecationWarning):
        assert_equal(np.fromstring('1. 2. 3. flop 4.', dtype=float, sep=' '), np.array([1.0, 2.0, 3.0]))