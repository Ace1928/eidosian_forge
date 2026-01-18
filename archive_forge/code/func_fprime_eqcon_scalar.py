from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def fprime_eqcon_scalar(self, x, sign=1.0):
    """ Scalar equality constraint, derivative """
    return self.fprime_eqcon(x, sign)[0].tolist()