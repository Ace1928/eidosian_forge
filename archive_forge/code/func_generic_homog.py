import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
def generic_homog(self, args, kws):
    if args:
        raise NumbaAssertionError('args not supported')
    if kws:
        raise NumbaAssertionError('kws not supported')
    return signature(self.this.dtype, recvr=self.this)