import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def jac2(t, x, omega1, omega2):
    j = array([[0.0, omega1], [-omega2, 0.0]])
    return j