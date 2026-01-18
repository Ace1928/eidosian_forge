from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def test_broyden1_update(self):
    jac = nonlin.BroydenFirst(alpha=0.1)
    jac.setup(self.xs[0], self.fs[0], None)
    B = np.identity(5) * (-1 / 0.1)
    for last_j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
        df = f - self.fs[last_j]
        dx = x - self.xs[last_j]
        B += (df - dot(B, dx))[:, None] * dx[None, :] / dot(dx, dx)
        jac.update(x, f)
        assert_(np.allclose(jac.todense(), B, rtol=1e-10, atol=1e-13))