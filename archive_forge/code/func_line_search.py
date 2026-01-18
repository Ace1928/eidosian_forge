import time
import warnings
from math import sqrt
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.constraints import UnitCellFilter
from ase.utils.linesearch import LineSearch
from ase.utils.linesearcharmijo import LineSearchArmijo
from ase.optimize.precon.precon import make_precon
def line_search(self, r, g, e, previously_reset_hessian):
    self.p = self.p.ravel()
    p_size = np.sqrt((self.p ** 2).sum())
    if p_size <= np.sqrt(len(self.atoms) * 1e-10):
        self.p /= p_size / np.sqrt(len(self.atoms) * 1e-10)
    g = g.ravel()
    r = r.ravel()
    if self.use_armijo:
        try:
            ls = LineSearchArmijo(self.func, c1=self.c1, tol=1e-14)
            step, func_val, no_update = ls.run(r, self.p, a_min=self.a_min, func_start=e, func_prime_start=g, func_old=self.e0, rigid_units=self.rigid_units, rotation_factors=self.rotation_factors, maxstep=self.maxstep)
            self.e0 = e
            self.e1 = func_val
            self.alpha_k = step
        except (ValueError, RuntimeError):
            if not previously_reset_hessian:
                warnings.warn('Armijo linesearch failed, resetting Hessian and trying again')
                self.reset_hessian()
                self.alpha_k = 0.0
            else:
                raise RuntimeError('Armijo linesearch failed after reset of Hessian, aborting')
    else:
        ls = LineSearch()
        self.alpha_k, e, self.e0, self.no_update = ls._line_search(self.func, self.fprime, r, self.p, g, e, self.e0, stpmin=self.a_min, maxstep=self.maxstep, c1=self.c1, c2=self.c2, stpmax=50.0)
        self.e1 = e
        if self.alpha_k is None:
            raise RuntimeError('Wolff lineSearch failed!')