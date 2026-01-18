from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def gen_epilogue(self, builder, pyapi):
    pyapi.gil_release(self.gil)