from collections import defaultdict
import copy
import sys
from itertools import permutations, takewhile
from contextlib import contextmanager
from functools import cached_property
from llvmlite import ir as llvmir
from llvmlite.ir import Constant
import llvmlite.binding as ll
from numba.core import types, utils, datamodel, debuginfo, funcdesc, config, cgutils, imputils
from numba.core import event, errors, targetconfig
from numba import _dynfunc, _helperlib
from numba.core.compiler_lock import global_compiler_lock
from numba.core.pythonapi import PythonAPI
from numba.core.imputils import (user_function, user_generator,
from numba.cpython import builtins
def compile_subroutine(self, builder, impl, sig, locals={}, flags=None, caching=True):
    """
        Compile the function *impl* for the given *sig* (in nopython mode).
        Return an instance of CompileResult.

        If *caching* evaluates True, the function keeps the compiled function
        for reuse in *.cached_internal_func*.
        """
    cache_key = (impl.__code__, sig, type(self.error_model))
    if not caching:
        cached = None
    else:
        if impl.__closure__:
            cache_key += tuple((c.cell_contents for c in impl.__closure__))
        cached = self.cached_internal_func.get(cache_key)
    if cached is None:
        cres = self._compile_subroutine_no_cache(builder, impl, sig, locals=locals, flags=flags)
        self.cached_internal_func[cache_key] = cres
    cres = self.cached_internal_func[cache_key]
    self.active_code_library.add_linking_library(cres.library)
    return cres