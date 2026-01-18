import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
def rop_real(s, t):
    ctx = s.ctx
    if not isinstance(t, ctx._types):
        t = ctx.convert(t)
    if hasattr(t, '_mpi_'):
        return g_real(ctx, t._mpi_, s._mpi_)
    if hasattr(t, '_mpci_'):
        return g_complex(ctx, t._mpci_, (s._mpi_, mpi_zero))
    return NotImplemented