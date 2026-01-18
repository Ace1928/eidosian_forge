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
@pytest.mark.xslow
def test_gh_1766():
    size = 22
    kx, ky = (3, 3)

    def f2(x, y):
        return np.sin(x + y)
    x = np.linspace(0, 10, size)
    y = np.linspace(50, 700, size)
    xy = makepairs(x, y)
    tck = bisplrep(xy[0], xy[1], f2(xy[0], xy[1]), s=0, kx=kx, ky=ky)
    tx_ty_size = 500000
    tck[0] = np.arange(tx_ty_size)
    tck[1] = np.arange(tx_ty_size) * 4
    tt_0 = np.arange(50)
    tt_1 = np.arange(50) * 3
    with pytest.raises(MemoryError):
        bisplev(tt_0, tt_1, tck, 1, 1)