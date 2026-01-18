import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
def resolve_kind(self, ary):
    if isinstance(ary.key, types.scalars.Float):
        val = 'f'
    elif isinstance(ary.key, types.scalars.Integer):
        val = 'i'
    else:
        return None
    return types.StringLiteral(val)