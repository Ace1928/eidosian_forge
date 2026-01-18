import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@bound_function('slice.indices')
def resolve_indices(self, ty, args, kws):
    assert not kws
    if len(args) != 1:
        raise errors.NumbaTypeError('indices() takes exactly one argument (%d given)' % len(args))
    typ, = args
    if not isinstance(typ, types.Integer):
        raise errors.NumbaTypeError("'%s' object cannot be interpreted as an integer" % typ)
    return signature(types.UniTuple(types.intp, 3), types.intp)