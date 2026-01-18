from .libmp.backend import basestring, exec_
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import rational
from . import function_docs
@classmethod
def mpf_convert_arg(cls, x, prec, rounding):
    if isinstance(x, int_types):
        return from_int(x)
    if isinstance(x, float):
        return from_float(x)
    if isinstance(x, basestring):
        return from_str(x, prec, rounding)
    if isinstance(x, cls.context.constant):
        return x.func(prec, rounding)
    if hasattr(x, '_mpf_'):
        return x._mpf_
    if hasattr(x, '_mpmath_'):
        t = cls.context.convert(x._mpmath_(prec, rounding))
        if hasattr(t, '_mpf_'):
            return t._mpf_
    if hasattr(x, '_mpi_'):
        a, b = x._mpi_
        if a == b:
            return a
        raise ValueError('can only create mpf from zero-width interval')
    raise TypeError('cannot create mpf from ' + repr(x))