from itertools import product
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.spatial.transform._rotation_spline import (
def test_angular_rate_nonlinear_term():
    np.random.seed(0)
    rv = np.random.rand(4, 3)
    assert_allclose(_angular_acceleration_nonlinear_term(rv, rv), 0, atol=1e-19)