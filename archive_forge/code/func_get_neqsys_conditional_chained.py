import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def get_neqsys_conditional_chained(self, rref_equil=False, rref_preserv=False, NumSys=NumSysLin, **kwargs):
    from pyneqsys import ConditionalNeqSys, ChainedNeqSys

    def factory(conds):
        return ChainedNeqSys([self._SymbolicSys_from_NumSys(NS, conds, rref_equil, rref_preserv, **kwargs) for NS in NumSys])
    cond_cbs = [(self._fw_cond_factory(ri), self._bw_cond_factory(ri, NumSys[0].small)) for ri in self.phase_transfer_reaction_idxs()]
    return ConditionalNeqSys(cond_cbs, factory)