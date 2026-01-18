import copy
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import cos, sin
from scipy.optimize import basinhopping, OptimizeResult
from scipy.optimize._basinhopping import (
def func2d_easyderiv(x):
    f = 2.0 * x[0] ** 2 + 2.0 * x[0] * x[1] + 2.0 * x[1] ** 2 - 6.0 * x[0]
    df = np.zeros(2)
    df[0] = 4.0 * x[0] + 2.0 * x[1] - 6.0
    df[1] = 2.0 * x[0] + 4.0 * x[1]
    return (f, df)