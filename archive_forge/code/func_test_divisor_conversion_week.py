import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_divisor_conversion_week(self):
    assert_(np.dtype('m8[W/7]') == np.dtype('m8[D]'))
    assert_(np.dtype('m8[3W/14]') == np.dtype('m8[36h]'))
    assert_(np.dtype('m8[5W/140]') == np.dtype('m8[360m]'))