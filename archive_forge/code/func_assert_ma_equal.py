import warnings
import numpy as np
import pytest
from numpy.core import finfo, iinfo
from numpy import half, single, double, longdouble
from numpy.testing import assert_equal, assert_, assert_raises
from numpy.core.getlimits import _discovered_machar, _float_ma
def assert_ma_equal(discovered, ma_like):
    for key, value in discovered.__dict__.items():
        assert_equal(value, getattr(ma_like, key))
        if hasattr(value, 'shape'):
            assert_equal(value.shape, getattr(ma_like, key).shape)
            assert_equal(value.dtype, getattr(ma_like, key).dtype)