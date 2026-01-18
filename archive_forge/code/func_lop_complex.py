import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
def lop_complex(s, t):
    if isinstance(t, _matrix):
        return NotImplemented
    ctx = s.ctx
    if not isinstance(t, s.ctx._types):
        try:
            t = s.ctx.convert(t)
        except (ValueError, TypeError):
            return NotImplemented
    return g_complex(ctx, s._mpci_, t._mpci_)