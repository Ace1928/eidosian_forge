from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def f_ieqcon(self, x, sign=1.0):
    """ Inequality constraint """
    return np.array([x[0] - x[1] - 1.0])