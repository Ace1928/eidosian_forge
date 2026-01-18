from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def compute_pri_res(self, x, z):
    """
        Compute primal residual ||Ax - z||
        """
    Ax = self.work.data.A.dot(x)
    pri_res = Ax - z
    if self.work.settings.scaling and (not self.work.settings.scaled_termination):
        pri_res = self.work.scaling.Einv.dot(pri_res)
    return la.norm(pri_res, np.inf)