import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
@pytest.mark.parametrize('bool_val', [True, False])
def test_longdouble_from_bool(bool_val):
    assert np.longdouble(bool_val) == np.longdouble(int(bool_val))