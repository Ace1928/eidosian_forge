import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def solout(t, y):
    ts.append(t)
    ys.append(y.copy())
    if t > tend / 2.0:
        return -1