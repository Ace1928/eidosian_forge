import warnings
import numpy as np
import pytest
from numpy.core import finfo, iinfo
from numpy import half, single, double, longdouble
from numpy.testing import assert_equal, assert_, assert_raises
from numpy.core.getlimits import _discovered_machar, _float_ma
def test_finfo_repr(self):
    expected = 'finfo(resolution=1e-06, min=-3.4028235e+38,' + ' max=3.4028235e+38, dtype=float32)'
    assert_equal(repr(np.finfo(np.float32)), expected)