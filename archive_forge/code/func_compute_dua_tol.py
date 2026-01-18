from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def compute_dua_tol(self, eps_abs, eps_rel):
    """
        Compute dual tolerance
        """
    P = self.work.data.P
    q = self.work.data.q
    A = self.work.data.A
    if self.work.settings.scaling and (not self.work.settings.scaled_termination):
        cinv = self.work.scaling.cinv
        Dinv = self.work.scaling.Dinv
        max_rel_eps = cinv * np.max([la.norm(Dinv.dot(A.T.dot(self.work.y)), np.inf), la.norm(Dinv.dot(P.dot(self.work.x)), np.inf), la.norm(Dinv.dot(q), np.inf)])
    else:
        max_rel_eps = np.max([la.norm(A.T.dot(self.work.y), np.inf), la.norm(P.dot(self.work.x), np.inf), la.norm(q, np.inf)])
    eps_dua = eps_abs + eps_rel * max_rel_eps
    return eps_dua