from scipy.constants import find, value, ConstantWarning, c, speed_of_light
from numpy.testing import (assert_equal, assert_, assert_almost_equal,
import scipy.constants._codata as _cd
def test_2002_vs_2006():
    assert_almost_equal(value('magn. flux quantum'), value('mag. flux quantum'))