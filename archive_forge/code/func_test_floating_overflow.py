import pytest
import numpy as np
from numpy.testing import (
def test_floating_overflow(self):
    """ Strings containing an unrepresentable float overflow """
    fhalf = np.half('1e10000')
    assert_equal(fhalf, np.inf)
    fsingle = np.single('1e10000')
    assert_equal(fsingle, np.inf)
    fdouble = np.double('1e10000')
    assert_equal(fdouble, np.inf)
    flongdouble = assert_warns(RuntimeWarning, np.longdouble, '1e10000')
    assert_equal(flongdouble, np.inf)
    fhalf = np.half('-1e10000')
    assert_equal(fhalf, -np.inf)
    fsingle = np.single('-1e10000')
    assert_equal(fsingle, -np.inf)
    fdouble = np.double('-1e10000')
    assert_equal(fdouble, -np.inf)
    flongdouble = assert_warns(RuntimeWarning, np.longdouble, '-1e10000')
    assert_equal(flongdouble, -np.inf)