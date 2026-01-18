import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
def generic_expand_cumulative(self, args, kws):
    if args:
        raise NumbaAssertionError('args unsupported')
    if kws:
        raise NumbaAssertionError('kwargs unsupported')
    assert isinstance(self.this, types.Array)
    return_type = types.Array(dtype=_expand_integer(self.this.dtype), ndim=1, layout='C')
    return signature(return_type, recvr=self.this)