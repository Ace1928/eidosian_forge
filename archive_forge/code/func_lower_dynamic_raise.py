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
def lower_dynamic_raise(self, inst):
    exc_args = inst.exc_args
    args = []
    nb_types = []
    for exc_arg in exc_args:
        if isinstance(exc_arg, ir.Var):
            typ = self.typeof(exc_arg.name)
            val = self.loadvar(exc_arg.name)
            self.incref(typ, val)
        else:
            typ = None
            val = exc_arg
        nb_types.append(typ)
        args.append(val)
    self.return_dynamic_exception(inst.exc_class, tuple(args), tuple(nb_types), loc=self.loc)