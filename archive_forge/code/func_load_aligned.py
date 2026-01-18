from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def load_aligned(self, ind):
    ptr = self.builder.gep(self.dataptr, [ind])
    return self.context.unpack_value(self.builder, self.fe_type, ptr)