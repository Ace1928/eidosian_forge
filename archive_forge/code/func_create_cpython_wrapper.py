from collections import namedtuple, defaultdict
import operator
import warnings
from functools import partial
import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import (typing, utils, types, ir, debuginfo, funcdesc,
from numba.core.errors import (LoweringError, new_error_context, TypingError,
from numba.core.funcdesc import default_mangler
from numba.core.environment import Environment
from numba.core.analysis import compute_use_defs, must_use_alloca
from numba.misc.firstlinefinder import get_func_body_first_lineno
def create_cpython_wrapper(self, release_gil=False):
    """
        Create CPython wrapper(s) around this function (or generator).
        """
    if self.genlower:
        self.context.create_cpython_wrapper(self.library, self.genlower.gendesc, self.env, self.call_helper, release_gil=release_gil)
    self.context.create_cpython_wrapper(self.library, self.fndesc, self.env, self.call_helper, release_gil=release_gil)