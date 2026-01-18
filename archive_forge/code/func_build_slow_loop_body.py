from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def build_slow_loop_body(context, func, builder, arrays, out, offsets, store_offset, signature, pyapi, env):

    def load():
        elems = [ary.load_direct(builder.load(off)) for off, ary in zip(offsets, arrays)]
        return elems

    def store(retval):
        out.store_direct(retval, builder.load(store_offset))
    return _build_ufunc_loop_body(load, store, context, func, builder, arrays, out, offsets, store_offset, signature, pyapi, env=env)