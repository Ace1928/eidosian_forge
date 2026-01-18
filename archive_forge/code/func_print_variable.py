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
def print_variable(self, msg, varname):
    """Helper to emit ``print(msg, varname)`` for debugging.

        Parameters
        ----------
        msg : str
            Literal string to be printed.
        varname : str
            A variable name whose value will be printed.
        """
    argtys = (types.literal(msg), self.fndesc.typemap[varname])
    args = (self.context.get_dummy_value(), self.loadvar(varname))
    sig = typing.signature(types.none, *argtys)
    impl = self.context.get_function(print, sig)
    impl(self.builder, args)