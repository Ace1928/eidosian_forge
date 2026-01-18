from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def compute_dua_res(self, x, y):
    """
        Compute dual residual ||Px + q + A'y||
        """
    dua_res = self.work.data.P.dot(x) + self.work.data.q + self.work.data.A.T.dot(y)
    if self.work.settings.scaling and (not self.work.settings.scaled_termination):
        dua_res = self.work.scaling.cinv * self.work.scaling.Dinv.dot(dua_res)
    return la.norm(dua_res, np.inf)