import math
import random
import numpy as np
from llvmlite import ir
from numba.core.cgutils import is_nonelike
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core.imputils import (Registry, impl_ret_untracked,
from numba.core.typing import signature
from numba.core import types, cgutils
from numba.np import arrayobj
from numba.core.errors import NumbaTypeError
def get_state_ptr(context, builder, name):
    """
    Get a pointer to the given thread-local random state
    (depending on *name*: "py" or "np").
    If the state isn't initialized, it is lazily initialized with
    system entropy.
    """
    assert name in ('py', 'np', 'internal')
    func_name = 'numba_get_%s_random_state' % name
    fnty = ir.FunctionType(rnd_state_ptr_t, ())
    fn = cgutils.get_or_insert_function(builder.module, fnty, func_name)
    fn.attributes.add('readnone')
    fn.attributes.add('nounwind')
    return builder.call(fn, ())