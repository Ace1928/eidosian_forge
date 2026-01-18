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
def lower_getitem(self, resty, expr, value, index, signature):
    baseval = self.loadvar(value.name)
    indexval = self.loadvar(index.name)
    op = operator.getitem
    fnop = self.context.typing_context.resolve_value_type(op)
    callsig = fnop.get_call_type(self.context.typing_context, signature.args, {})
    impl = self.context.get_function(fnop, callsig)
    argvals = (baseval, indexval)
    argtyps = (self.typeof(value.name), self.typeof(index.name))
    castvals = [self.context.cast(self.builder, av, at, ft) for av, at, ft in zip(argvals, argtyps, signature.args)]
    res = impl(self.builder, castvals)
    return self.context.cast(self.builder, res, signature.return_type, resty)