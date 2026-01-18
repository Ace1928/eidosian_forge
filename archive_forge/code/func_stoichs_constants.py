import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def stoichs_constants(self, eq_params=None, rref=False, Matrix=None, backend=None, non_precip_rids=()):
    if eq_params is None:
        eq_params = self.eq_constants()
    if rref:
        from pyneqsys.symbolic import linear_rref
        be = get_backend(backend)
        rA, rb = linear_rref(self.stoichs(non_precip_rids), list(map(be.log, eq_params)), Matrix)
        return (rA.tolist(), list(map(be.exp, rb)))
    else:
        return (self.stoichs(non_precip_rids), eq_params)