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
def test_extrapolation_modes(self):
    x = [1, 2, 3]
    y = [0, 2, 4]
    tck = splrep(x, y, k=1)
    rstl = [[-2, 6], [0, 0], None, [0, 4]]
    for ext in (0, 1, 3):
        assert_array_almost_equal(splev([0, 4], tck, ext=ext), rstl[ext])
    assert_raises(ValueError, splev, [0, 4], tck, ext=2)