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
def fold_call_args(self, fnty, signature, pos_args, vararg, kw_args):
    if vararg:
        tp_vararg = self.typeof(vararg.name)
        assert isinstance(tp_vararg, types.BaseTuple)
        pos_args = pos_args + [_VarArgItem(vararg, i) for i in range(len(tp_vararg))]
    pysig = signature.pysig
    if pysig is None:
        if kw_args:
            raise NotImplementedError('unsupported keyword arguments when calling %s' % (fnty,))
        argvals = [self._cast_var(var, sigty) for var, sigty in zip(pos_args, signature.args)]
    else:

        def normal_handler(index, param, var):
            return self._cast_var(var, signature.args[index])

        def default_handler(index, param, default):
            return self.context.get_constant_generic(self.builder, signature.args[index], default)

        def stararg_handler(index, param, vars):
            stararg_ty = signature.args[index]
            assert isinstance(stararg_ty, types.BaseTuple), stararg_ty
            values = [self._cast_var(var, sigty) for var, sigty in zip(vars, stararg_ty)]
            return cgutils.make_anonymous_struct(self.builder, values)
        argvals = typing.fold_arguments(pysig, pos_args, dict(kw_args), normal_handler, default_handler, stararg_handler)
    return argvals