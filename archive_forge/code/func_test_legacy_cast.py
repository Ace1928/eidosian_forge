from typing import Callable
import numpy as np
from numpy.testing import assert_array_equal, assert_, suppress_warnings
import pytest
import scipy.special as sc
def test_legacy_cast():
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'floating point number truncated to an integer')
        res = sc.bdtrc(np.nan, 1, 0.5)
        assert_(np.isnan(res))