import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_fromstring_missing():
    with assert_warns(DeprecationWarning):
        assert_equal(np.fromstring('1xx3x4x5x6', sep='x'), np.array([1]))