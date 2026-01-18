import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def composition_conservation(self, concs, init_concs):
    composition_vecs, comp_keys = self.composition_balance_vectors()
    A = np.array(composition_vecs)
    return (comp_keys, np.dot(A, self.as_per_substance_array(concs).T), np.dot(A, self.as_per_substance_array(init_concs).T))