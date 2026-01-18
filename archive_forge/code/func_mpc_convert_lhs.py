from .libmp.backend import basestring, exec_
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import rational
from . import function_docs
@classmethod
def mpc_convert_lhs(cls, x):
    try:
        y = cls.context.convert(x)
        return y
    except TypeError:
        return NotImplemented