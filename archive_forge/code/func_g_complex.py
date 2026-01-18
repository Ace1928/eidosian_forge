import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
def g_complex(ctx, sval, tval):
    return ctx.make_mpc(f_complex(sval, tval, ctx.prec))