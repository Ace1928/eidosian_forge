from .libmp.backend import basestring, exec_
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import rational
from . import function_docs
def make_mpf(ctx, v):
    a = new(ctx.mpf)
    a._mpf_ = v
    return a