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
@register_jitable
def raise_if_incompatible_array_sizes(a, *args):
    ashape = a.shape
    for arg in literal_unroll(args):
        if a.ndim != arg.ndim:
            raise ValueError('Secondary stencil array does not have same number  of dimensions as the first stencil input.')
        argshape = arg.shape
        for i in range(len(ashape)):
            if ashape[i] > argshape[i]:
                raise ValueError('Secondary stencil array has some dimension smaller the same dimension in the first stencil input.')