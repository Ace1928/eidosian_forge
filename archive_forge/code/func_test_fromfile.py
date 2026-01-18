import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
@pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
def test_fromfile(self):
    with temppath() as path:
        with open(path, 'w') as f:
            f.write(self.out)
        res = np.fromfile(path, dtype=np.longdouble, sep='\n')
    assert_equal(res, self.tgt)