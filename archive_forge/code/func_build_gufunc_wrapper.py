from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def build_gufunc_wrapper(py_func, cres, sin, sout, cache, is_parfors):
    signature = cres.signature
    wrapcls = _GufuncObjectWrapper if signature.return_type == types.pyobject else _GufuncWrapper
    return wrapcls(py_func, cres, sin, sout, cache, is_parfors=is_parfors).build()