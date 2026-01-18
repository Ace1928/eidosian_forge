import functools
import warnings
import numpy as np
from numba import jit, typeof
from numba.core import cgutils, types, serialize, sigutils, errors
from numba.core.extending import (is_jitted, overload_attribute,
from numba.core.typing import npydecl
from numba.core.typing.templates import AbstractTemplate, signature
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.ufunc import _internal
from numba.parfors import array_analysis
from numba.np.ufunc import ufuncbuilder
from numba.np import numpy_support
from typing import Callable
from llvmlite import ir
def find_ewise_function(self, ewise_types):
    """
        Given a tuple of element-wise argument types, find a matching
        signature in the dispatcher.

        Return a 2-tuple containing the matching signature, and
        compilation result.  Will return two None's if no matching
        signature was found.
        """
    if self._frozen:
        loop = numpy_support.ufunc_find_matching_loop(self, ewise_types)
        if loop is None:
            return (None, None)
        ewise_types = tuple(loop.inputs + loop.outputs)[:len(ewise_types)]
    for sig, cres in self._dispatcher.overloads.items():
        if sig.args == ewise_types:
            return (sig, cres)
    return (None, None)