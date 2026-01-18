import warnings
import pytest
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
import numpy as np
from numpy import array, float64
from multiprocessing.pool import ThreadPool
from scipy import optimize, linalg
from scipy.special import lambertw
from scipy.optimize._minpack_py import leastsq, curve_fit, fixed_point
from scipy.optimize import OptimizeWarning
from scipy.optimize._minimize import Bounds
def hyperbola(x, s_1, s_2, o_x, o_y, c):
    b_2 = (s_1 + s_2) / 2
    b_1 = (s_2 - s_1) / 2
    return o_y + b_1 * (x - o_x) + b_2 * np.sqrt((x - o_x) ** 2 + c ** 2 / 4)