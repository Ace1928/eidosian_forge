import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@bound_function('array.argsort')
def resolve_argsort(self, ary, args, kws):
    assert not args
    kwargs = dict(kws)
    kind = kwargs.pop('kind', types.StringLiteral('quicksort'))
    if not isinstance(kind, types.StringLiteral):
        raise TypingError('"kind" must be a string literal')
    if kwargs:
        msg = 'Unsupported keywords: {!r}'
        raise TypingError(msg.format([k for k in kwargs.keys()]))
    if ary.ndim == 1:

        def argsort_stub(kind='quicksort'):
            pass
        pysig = utils.pysignature(argsort_stub)
        sig = signature(types.Array(types.intp, 1, 'C'), kind).replace(pysig=pysig)
        return sig