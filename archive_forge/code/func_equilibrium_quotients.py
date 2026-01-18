import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def equilibrium_quotients(self, concs):
    stoichs = self.stoichs()
    return [equilibrium_quotient(concs, stoichs[ri, :]) for ri in range(self.nr)]