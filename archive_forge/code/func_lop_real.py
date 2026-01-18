import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
def lop_real(s, t):
    if isinstance(t, _matrix):
        return NotImplemented
    ctx = s.ctx
    if not isinstance(t, ctx._types):
        t = ctx.convert(t)
    if hasattr(t, '_mpi_'):
        return g_real(ctx, s._mpi_, t._mpi_)
    if hasattr(t, '_mpci_'):
        return g_complex(ctx, (s._mpi_, mpi_zero), t._mpci_)
    return NotImplemented