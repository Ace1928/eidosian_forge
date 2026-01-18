import itertools
import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_,
from pytest import raises as assert_raises
import pytest
from scipy._lib._testutils import check_free_memory
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate._fitpack_py import (splrep, splev, bisplrep, bisplev,
from scipy.interpolate.dfitpack import regrid_smth
from scipy.interpolate._fitpack2 import dfitpack_int
def test_dblint():
    x = np.linspace(0, 1)
    y = np.linspace(0, 1)
    xx, yy = np.meshgrid(x, y)
    rect = RectBivariateSpline(x, y, 4 * xx * yy)
    tck = list(rect.tck)
    tck.extend(rect.degrees)
    assert_almost_equal(dblint(0, 1, 0, 1, tck), 1)
    assert_almost_equal(dblint(0, 0.5, 0, 1, tck), 0.25)
    assert_almost_equal(dblint(0.5, 1, 0, 1, tck), 0.75)
    assert_almost_equal(dblint(-100, 100, -100, 100, tck), 1)