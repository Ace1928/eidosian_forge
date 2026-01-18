import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@bound_function('array.view')
def resolve_view(self, ary, args, kws):
    from .npydecl import parse_dtype
    assert not kws
    dtype, = args
    dtype = parse_dtype(dtype)
    if dtype is None:
        return
    retty = ary.copy(dtype=dtype)
    return signature(retty, *args)