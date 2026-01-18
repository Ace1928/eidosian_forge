from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def store_solution(self):
    """
        Store current primal and dual solution in solution structure
        """
    if self.work.info.status_val is not OSQP_PRIMAL_INFEASIBLE and self.work.info.status_val is not OSQP_DUAL_INFEASIBLE:
        self.work.solution.x = self.work.x
        self.work.solution.y = self.work.y
        if self.work.settings.scaling:
            self.work.solution.x = self.work.scaling.D.dot(self.work.solution.x)
            self.work.solution.y = self.work.scaling.cinv * self.work.scaling.E.dot(self.work.solution.y)
    else:
        self.work.solution.x = np.array([None] * self.work.data.n)
        self.work.solution.y = np.array([None] * self.work.data.m)