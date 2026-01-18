from __future__ import absolute_import
import types
import warnings
from autograd.extend import primitive, notrace_primitive
import numpy as _np
import autograd.builtins as builtins
from numpy.core.einsumfunc import _parse_einsum_input
@primitive
def make_diagonal(D, offset=0, axis1=0, axis2=1):
    if not (offset == 0 and axis1 == -1 and (axis2 == -2)):
        raise NotImplementedError('Currently make_diagonal only supports offset=0, axis1=-1, axis2=-2')
    new_array = _np.zeros(D.shape + (D.shape[-1],))
    new_array_diag = _np.diagonal(new_array, offset=0, axis1=-1, axis2=-2)
    new_array_diag.flags.writeable = True
    new_array_diag[:] = D
    return new_array