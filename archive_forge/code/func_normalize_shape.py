import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
def normalize_shape(shape):
    if isinstance(shape, types.UniTuple):
        if isinstance(shape.dtype, types.Integer):
            dimtype = types.intp if shape.dtype.signed else types.uintp
            return types.UniTuple(dimtype, len(shape))
    elif isinstance(shape, types.Tuple) and shape.count == 0:
        return types.UniTuple(types.intp, 0)