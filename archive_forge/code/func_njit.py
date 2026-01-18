import sys
import warnings
import inspect
import logging
from numba.core.errors import DeprecationError, NumbaDeprecationWarning
from numba.stencils.stencil import stencil
from numba.core import config, extending, sigutils, registry
def njit(*args, **kws):
    """
    Equivalent to jit(nopython=True)

    See documentation for jit function/decorator for full description.
    """
    if 'nopython' in kws:
        warnings.warn('nopython is set for njit and is ignored', RuntimeWarning)
    if 'forceobj' in kws:
        warnings.warn('forceobj is set for njit and is ignored', RuntimeWarning)
        del kws['forceobj']
    kws.update({'nopython': True})
    return jit(*args, **kws)