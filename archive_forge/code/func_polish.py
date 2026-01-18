from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def polish(self):
    """
        Solution polish:
        Solve equality constrained QP with assumed active constraints.
        """
    self.work.timer = time.time()
    self.work.pol.ind_low = np.where(self.work.z - self.work.data.l < -self.work.y)[0]
    self.work.pol.ind_upp = np.where(self.work.data.u - self.work.z < self.work.y)[0]
    self.work.pol.n_low = len(self.work.pol.ind_low)
    self.work.pol.n_upp = len(self.work.pol.ind_upp)
    self.work.pol.Ared = spspa.vstack([self.work.data.A[self.work.pol.ind_low], self.work.data.A[self.work.pol.ind_upp]])
    KKTred = spspa.vstack([spspa.hstack([self.work.data.P + self.work.settings.delta * spspa.eye(self.work.data.n), self.work.pol.Ared.T]), spspa.hstack([self.work.pol.Ared, -self.work.settings.delta * spspa.eye(self.work.pol.Ared.shape[0])])])
    KKTred_factor = spla.splu(KKTred.tocsc())
    rhs_red = np.hstack([-self.work.data.q, self.work.data.l[self.work.pol.ind_low], self.work.data.u[self.work.pol.ind_upp]])
    pol_sol = KKTred_factor.solve(rhs_red)
    if self.work.settings.polish_refine_iter > 0:
        pol_sol = self.iter_refin(KKTred_factor, pol_sol, rhs_red)
    self.work.pol.x = pol_sol[:self.work.data.n]
    self.work.pol.z = self.work.data.A.dot(self.work.pol.x)
    self.work.pol.y = np.zeros(self.work.data.m)
    y_red = pol_sol[self.work.data.n:]
    self.work.pol.y[self.work.pol.ind_low] = y_red[:self.work.pol.n_low]
    self.work.pol.y[self.work.pol.ind_upp] = y_red[self.work.pol.n_low:]
    self.work.pol.z, self.work.pol.y = self.project_normalcone(self.work.pol.z, self.work.pol.y)
    self.update_info(0, 1)
    pol_success = self.work.pol.pri_res < self.work.info.pri_res and self.work.pol.dua_res < self.work.info.dua_res or (self.work.pol.pri_res < self.work.info.pri_res and self.work.info.dua_res < 1e-10) or (self.work.pol.dua_res < self.work.info.dua_res and self.work.info.pri_res < 1e-10)
    ls = linesearch()
    if pol_success:
        self.work.info.obj_val = self.work.pol.obj_val
        self.work.info.pri_res = self.work.pol.pri_res
        self.work.info.dua_res = self.work.pol.dua_res
        self.work.info.status_polish = 1
        self.work.x = self.work.pol.x
        self.work.z = self.work.pol.z
        self.work.y = self.work.pol.y
        if self.work.settings.verbose:
            self.print_polish()
    else:
        self.work.info.status_polish = -1
        ls.t = np.linspace(0.0, 0.002, 1000)
        ls.X, ls.Z, ls.Y = self.line_search(self.work.x, self.work.z, self.work.y, self.work.pol.x, self.work.pol.z, self.work.pol.y, ls.t)
    return ls