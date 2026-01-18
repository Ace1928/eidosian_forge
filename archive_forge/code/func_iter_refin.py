from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def iter_refin(self, KKT_factor, z, b):
    """
        Iterative refinement of the solution of a linear system
            1. (K + dK) * dz = b - K*z
            2. z <- z + dz
        """
    for i in range(self.work.settings.polish_refine_iter):
        rhs = b - np.hstack([self.work.data.P.dot(z[:self.work.data.n]) + self.work.pol.Ared.T.dot(z[self.work.data.n:]), self.work.pol.Ared.dot(z[:self.work.data.n])])
        dz = KKT_factor.solve(rhs)
        z += dz
    return z