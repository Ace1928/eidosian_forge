from .libmp.backend import basestring, exec_
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import rational
from . import function_docs
@classmethod
def mpf_convert_lhs(cls, x):
    x = cls.mpf_convert_rhs(x)
    if type(x) is tuple:
        return cls.context.make_mpf(x)
    return x