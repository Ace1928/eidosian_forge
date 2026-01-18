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
def stencil(func_or_mode='constant', **options):
    if not isinstance(func_or_mode, str):
        mode = 'constant'
        func = func_or_mode
    else:
        mode = func_or_mode
        func = None
    for option in options:
        if option not in ['cval', 'standard_indexing', 'neighborhood']:
            raise ValueError('Unknown stencil option ' + option)
    wrapper = _stencil(mode, options)
    if func is not None:
        return wrapper(func)
    return wrapper