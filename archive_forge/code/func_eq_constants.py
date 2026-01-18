import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def eq_constants(self, non_precip_rids=(), eq_params=None, small=0):
    if eq_params is None:
        eq_params = [eq.param for eq in self.rxns]
    return [small if idx in non_precip_rids else eq for idx, eq in enumerate(eq_params)]