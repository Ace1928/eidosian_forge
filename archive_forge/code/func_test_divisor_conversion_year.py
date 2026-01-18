import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_divisor_conversion_year(self):
    assert_(np.dtype('M8[Y/4]') == np.dtype('M8[3M]'))
    assert_(np.dtype('M8[Y/13]') == np.dtype('M8[4W]'))
    assert_(np.dtype('M8[3Y/73]') == np.dtype('M8[15D]'))