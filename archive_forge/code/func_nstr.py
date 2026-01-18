import operator
from . import libmp
from .libmp.backend import basestring
from .libmp import (
from .matrices.matrices import _matrix
from .ctx_base import StandardBaseContext
def nstr(ctx, x, n=5, **kwargs):
    x = ctx.convert(x)
    if hasattr(x, '_mpi_'):
        return libmp.mpi_to_str(x._mpi_, n, **kwargs)
    if hasattr(x, '_mpci_'):
        re = libmp.mpi_to_str(x._mpci_[0], n, **kwargs)
        im = libmp.mpi_to_str(x._mpci_[1], n, **kwargs)
        return '(%s + %s*j)' % (re, im)