import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def dissolved(self, concs):
    """Return dissolved concentrations"""
    new_concs = concs.copy()
    for r in self.rxns:
        if r.has_precipitates(self.substances):
            net_stoich = np.asarray(r.net_stoich(self.substances))
            s_net, s_stoich, s_idx = r.precipitate_stoich(self.substances)
            new_concs -= new_concs[s_idx] / s_stoich * net_stoich
    return new_concs