import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
def test_fromstring_foreign_repr(self):
    f = 1.234
    a = np.fromstring(repr(f), dtype=float, sep=' ')
    assert_equal(a[0], f)