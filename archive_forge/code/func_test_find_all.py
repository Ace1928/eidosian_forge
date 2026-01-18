from scipy.constants import find, value, ConstantWarning, c, speed_of_light
from numpy.testing import (assert_equal, assert_, assert_almost_equal,
import scipy.constants._codata as _cd
def test_find_all():
    assert_(len(find(disp=False)) > 300)