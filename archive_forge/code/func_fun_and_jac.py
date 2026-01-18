from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def fun_and_jac(self, d, sign=1.0):
    return (self.fun(d, sign), self.jac(d, sign))