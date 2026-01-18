from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def test_broyden2_update(self):
    jac = nonlin.BroydenSecond(alpha=0.1)
    jac.setup(self.xs[0], self.fs[0], None)
    H = np.identity(5) * -0.1
    for last_j, (x, f) in enumerate(zip(self.xs[1:], self.fs[1:])):
        df = f - self.fs[last_j]
        dx = x - self.xs[last_j]
        H += (dx - dot(H, df))[:, None] * df[None, :] / dot(df, df)
        jac.update(x, f)
        assert_(np.allclose(jac.todense(), inv(H), rtol=1e-10, atol=1e-13))