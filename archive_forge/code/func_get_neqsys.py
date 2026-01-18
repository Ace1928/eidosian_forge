import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
def get_neqsys(self, neqsys_type, NumSys=NumSysLin, **kwargs):
    new_kw = {'rref_equil': False, 'rref_preserv': False}
    if neqsys_type == 'static_conditions':
        new_kw['precipitates'] = None
    for k in new_kw:
        if k in kwargs:
            new_kw[k] = kwargs.pop(k)
    try:
        NumSys[0]
    except TypeError:
        new_kw['NumSys'] = (NumSys,)
    else:
        new_kw['NumSys'] = NumSys
    return getattr(self, 'get_neqsys_' + neqsys_type)(**new_kw)