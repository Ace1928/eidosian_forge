from .libmp.backend import basestring, exec_
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import rational
from . import function_docs
@classmethod
def mpf_convert_rhs(cls, x):
    if isinstance(x, int_types):
        return from_int(x)
    if isinstance(x, float):
        return from_float(x)
    if isinstance(x, complex_types):
        return cls.context.mpc(x)
    if isinstance(x, rational.mpq):
        p, q = x._mpq_
        return from_rational(p, q, cls.context.prec)
    if hasattr(x, '_mpf_'):
        return x._mpf_
    if hasattr(x, '_mpmath_'):
        t = cls.context.convert(x._mpmath_(*cls.context._prec_rounding))
        if hasattr(t, '_mpf_'):
            return t._mpf_
        return t
    return NotImplemented