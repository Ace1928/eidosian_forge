import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
def inline_function(self, caller_ir, block, i, function, arg_typs=None):
    """ Inlines the function in the caller_ir at statement index i of block
        `block`. If `arg_typs` is given and the InlineWorker instance was
        initialized with a typemap and calltypes then they will be appropriately
        updated based on the arg_typs.
        """
    callee_ir = self.run_untyped_passes(function)
    freevars = function.__code__.co_freevars
    return self.inline_ir(caller_ir, block, i, callee_ir, freevars, arg_typs=arg_typs)