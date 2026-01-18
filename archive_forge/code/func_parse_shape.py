import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
def parse_shape(shape):
    """
    Given a shape, return the number of dimensions.
    """
    ndim = None
    if isinstance(shape, types.Integer):
        ndim = 1
    elif isinstance(shape, (types.Tuple, types.UniTuple)):
        int_tys = (types.Integer, types.IntEnumMember)
        if all((isinstance(s, int_tys) for s in shape)):
            ndim = len(shape)
    return ndim