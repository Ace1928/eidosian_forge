import abc
from contextlib import contextmanager
from collections import defaultdict, namedtuple
from functools import partial
from copy import copy
import warnings
from numba.core import (errors, types, typing, ir, funcdesc, rewrites,
from numba.parfors.parfor import PreParforPass as _parfor_PreParforPass
from numba.parfors.parfor import ParforPass as _parfor_ParforPass
from numba.parfors.parfor import ParforFusionPass as _parfor_ParforFusionPass
from numba.parfors.parfor import ParforPreLoweringPass as \
from numba.parfors.parfor import Parfor
from numba.parfors.parfor_lowering import ParforLower
from numba.core.compiler_machinery import (FunctionPass, LoweringPass,
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (raise_on_unsupported_feature, warn_deprecated,
from numba.core import postproc
from llvmlite import binding as llvm
def get_func_type(state, expr):
    func_ty = None
    if expr.op == 'call':
        try:
            func_ty = state.typemap[expr.func.name]
        except KeyError:
            return None
        if not hasattr(func_ty, 'get_call_type'):
            return None
    elif is_operator_or_getitem(expr):
        func_ty = state.typingctx.resolve_value_type(expr.fn)
    else:
        return None
    return func_ty