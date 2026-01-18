import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def bjac_rows(y, t, c):
    jac = np.vstack((np.r_[0, np.diag(c, 1)], np.diag(c), np.r_[np.diag(c, -1), 0], np.r_[np.diag(c, -2), 0, 0]))
    return jac