import warnings
import numpy as np
import pytest
from numpy.core import finfo, iinfo
from numpy import half, single, double, longdouble
from numpy.testing import assert_equal, assert_, assert_raises
from numpy.core.getlimits import _discovered_machar, _float_ma
def assert_iinfo_equal(i1, i2):
    for attr in ('bits', 'min', 'max'):
        assert_equal(getattr(i1, attr), getattr(i2, attr), f'iinfo instances {i1} and {i2} differ on {attr}')