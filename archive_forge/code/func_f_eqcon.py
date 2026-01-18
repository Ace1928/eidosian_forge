from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def f_eqcon(x):
    """ Equality constraint """
    return x[1] - (2 * x[0]) ** 3