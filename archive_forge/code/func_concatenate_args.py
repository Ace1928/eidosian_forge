from __future__ import absolute_import
import types
import warnings
from autograd.extend import primitive, notrace_primitive
import numpy as _np
import autograd.builtins as builtins
from numpy.core.einsumfunc import _parse_einsum_input
@primitive
def concatenate_args(axis, *args):
    return _np.concatenate(args, axis).view(ndarray)