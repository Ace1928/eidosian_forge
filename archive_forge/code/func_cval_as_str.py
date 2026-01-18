import copy
import numpy as np
from llvmlite import ir as lir
from numba.core import types, typing, utils, ir, config, ir_utils, registry
from numba.core.typing.templates import (CallableTemplate, signature,
from numba.core.imputils import lower_builtin
from numba.core.extending import register_jitable
from numba.core.errors import NumbaValueError
from numba.misc.special import literal_unroll
import numba
import operator
from numba.np import numpy_support
def cval_as_str(cval):
    if not np.isfinite(cval):
        if np.isnan(cval):
            return 'np.nan'
        elif np.isinf(cval):
            if cval < 0:
                return '-np.inf'
            else:
                return 'np.inf'
    else:
        return str(cval)