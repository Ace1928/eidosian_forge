import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@bound_function('array.copy')
def resolve_copy(self, ary, args, kws):
    assert not args
    assert not kws
    retty = ary.copy(layout='C', readonly=False)
    return signature(retty)