import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import scipy.special as sc
from scipy.special._testutils import FuncData
def gammainc_line(self, x):
    c = np.array([-1 / 3, -1 / 540, 25 / 6048, 101 / 155520, -3184811 / 3695155200, -2745493 / 8151736420])
    res = 0
    xfac = 1
    for ck in c:
        res -= ck * xfac
        xfac /= x
    res /= np.sqrt(2 * np.pi * x)
    res += 0.5
    return res