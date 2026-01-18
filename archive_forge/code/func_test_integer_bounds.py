from numpy.testing import (assert_, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, Bounds, NonlinearConstraint
def test_integer_bounds(self):
    fmin_slsqp(lambda z: z ** 2 - 1, [0], bounds=[[0, 1]], iprint=0)