from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def build_fast_loop_body(context, func, builder, arrays, out, offsets, store_offset, signature, ind, pyapi, env):

    def load():
        elems = [ary.load_aligned(ind) for ary in arrays]
        return elems

    def store(retval):
        out.store_aligned(retval, ind)
    return _build_ufunc_loop_body(load, store, context, func, builder, arrays, out, offsets, store_offset, signature, pyapi, env=env)