import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_fromstring_foreign_sep(self):
    a = np.array([1, 2, 3, 4])
    b = np.fromstring('1,2,3,4,', dtype=np.longdouble, sep=',')
    assert_array_equal(a, b)