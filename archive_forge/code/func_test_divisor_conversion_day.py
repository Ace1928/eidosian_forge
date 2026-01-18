import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_divisor_conversion_day(self):
    assert_(np.dtype('M8[D/12]') == np.dtype('M8[2h]'))
    assert_(np.dtype('M8[D/120]') == np.dtype('M8[12m]'))
    assert_(np.dtype('M8[3D/960]') == np.dtype('M8[270s]'))