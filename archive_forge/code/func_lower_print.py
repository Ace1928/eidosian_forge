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
def lower_print(self, inst):
    """
        Lower a ir.Print()
        """
    sig = self.fndesc.calltypes[inst]
    assert sig.return_type == types.none
    fnty = self.context.typing_context.resolve_value_type(print)
    pos_tys = list(sig.args)
    pos_args = list(inst.args)
    for i in range(len(pos_args)):
        if i in inst.consts:
            pyval = inst.consts[i]
            if isinstance(pyval, str):
                pos_tys[i] = types.literal(pyval)
    fixed_sig = typing.signature(sig.return_type, *pos_tys)
    fixed_sig = fixed_sig.replace(pysig=sig.pysig)
    argvals = self.fold_call_args(fnty, sig, pos_args, inst.vararg, {})
    impl = self.context.get_function(print, fixed_sig)
    impl(self.builder, argvals)