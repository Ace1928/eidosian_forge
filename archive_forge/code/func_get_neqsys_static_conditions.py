import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def get_neqsys_static_conditions(self, rref_equil=False, rref_preserv=False, NumSys=(NumSysLin,), precipitates=None, **kwargs):
    if precipitates is None:
        precipitates = (False,) * len(self.phase_transfer_reaction_idxs())
    from pyneqsys import ChainedNeqSys
    return ChainedNeqSys([self._SymbolicSys_from_NumSys(NS, precipitates, rref_equil, rref_preserv, **kwargs) for NS in NumSys])