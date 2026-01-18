import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@bound_function('array.nonzero')
def resolve_nonzero(self, ary, args, kws):
    assert not args
    assert not kws
    ndim = max(ary.ndim, 1)
    retty = types.UniTuple(types.Array(types.intp, 1, 'C'), ndim)
    return signature(retty)