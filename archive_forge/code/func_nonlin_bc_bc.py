import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def nonlin_bc_bc(ya, yb):
    phiA, phipA = ya
    phiC, phipC = yb
    kappa, ioA, ioC, V, f = (1.64, 0.01, 0.0001, 0.5, 38.9)
    hA = 0.0 - phiA - 0.0
    iA = ioA * (np.exp(f * hA) - np.exp(-f * hA))
    res0 = iA + kappa * phipA
    hC = V - phiC - 1.0
    iC = ioC * (np.exp(f * hC) - np.exp(-f * hC))
    res1 = iC - kappa * phipC
    return np.array([res0, res1])