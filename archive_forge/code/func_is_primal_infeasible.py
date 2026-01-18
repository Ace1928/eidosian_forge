from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def is_primal_infeasible(self, eps_prim_inf):
    """
        Check primal infeasibility
                ||A'*v||_2 = 0
        with v = delta_y/||delta_y||_2 given that following condition holds
            u'*(v)_{+} + l'*(v)_{-} < 0
        """
    if self.work.settings.scaling and (not self.work.settings.scaled_termination):
        norm_delta_y = la.norm(self.work.scaling.E.dot(self.work.delta_y), np.inf)
    else:
        norm_delta_y = la.norm(self.work.delta_y, np.inf)
    if norm_delta_y > eps_prim_inf:
        lhs = self.work.data.u.dot(np.maximum(self.work.delta_y, 0)) + self.work.data.l.dot(np.minimum(self.work.delta_y, 0))
        if lhs < -eps_prim_inf * norm_delta_y:
            self.work.Atdelta_y = self.work.data.A.T.dot(self.work.delta_y)
            if self.work.settings.scaling and (not self.work.settings.scaled_termination):
                self.work.Atdelta_y = self.work.scaling.Dinv.dot(self.work.Atdelta_y)
            return la.norm(self.work.Atdelta_y, np.inf) < eps_prim_inf * norm_delta_y
    return False