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
def jacp(x, a, b):
    rotn = np.array([[1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0], [0, 0, 1.0]])
    e = np.exp(-b * x)
    return rotn.dot(np.vstack((e, -a * x * e)).T)