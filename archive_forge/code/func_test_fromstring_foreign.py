import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_fromstring_foreign(self):
    s = '1.234'
    a = np.fromstring(s, dtype=np.longdouble, sep=' ')
    assert_equal(a[0], np.longdouble(s))