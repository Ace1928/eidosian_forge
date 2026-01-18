import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@bound_function('array.ravel')
def resolve_ravel(self, ary, args, kws):
    assert not kws
    assert not args
    copy_will_be_made = ary.layout != 'C'
    readonly = not (copy_will_be_made or ary.mutable)
    return signature(ary.copy(ndim=1, layout='C', readonly=readonly))