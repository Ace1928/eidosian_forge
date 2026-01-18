import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
def g_real(ctx, sval, tval):
    try:
        return ctx.make_mpf(f_real(sval, tval, ctx.prec))
    except ComplexResult:
        sval = (sval, mpi_zero)
        tval = (tval, mpi_zero)
        return g_complex(ctx, sval, tval)