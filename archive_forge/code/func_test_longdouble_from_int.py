import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
@pytest.mark.parametrize('int_val', [2 ** 1024, 0])
def test_longdouble_from_int(int_val):
    str_val = str(int_val)
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', RuntimeWarning)
        assert np.longdouble(int_val) == np.longdouble(str_val)
        if np.allclose(np.finfo(np.longdouble).max, np.finfo(np.double).max) and w:
            assert w[0].category is RuntimeWarning