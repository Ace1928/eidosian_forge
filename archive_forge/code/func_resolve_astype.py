import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@bound_function('array.astype')
def resolve_astype(self, ary, args, kws):
    from .npydecl import parse_dtype
    assert not kws
    dtype, = args
    if isinstance(dtype, types.UnicodeType):
        raise RequireLiteralValue('array.astype if dtype is a string it must be constant')
    dtype = parse_dtype(dtype)
    if dtype is None:
        return
    if not self.context.can_convert(ary.dtype, dtype):
        raise TypeError('astype(%s) not supported on %s: cannot convert from %s to %s' % (dtype, ary, ary.dtype, dtype))
    layout = ary.layout if ary.layout in 'CF' else 'C'
    retty = ary.copy(dtype=dtype, layout=layout, readonly=False)
    return signature(retty, *args)