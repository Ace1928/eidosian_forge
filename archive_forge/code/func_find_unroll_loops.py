from collections import defaultdict, namedtuple
from contextlib import contextmanager
from copy import deepcopy, copy
import warnings
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core import (errors, types, ir, bytecode, postproc, rewrites, config,
from numba.misc.special import literal_unroll
from numba.core.analysis import (dead_branch_prune, rewrite_semantic_constants,
from numba.core.ir_utils import (guard, resolve_func_from_module, simplify_CFG,
from numba.core.ssa import reconstruct_ssa
from numba.core import interpreter
def find_unroll_loops(loops):
    """This finds loops which are compliant with the form:
            for i in range(len(literal_unroll(<something>>)))"""
    unroll_loops = {}
    for header_lbl, loop in loops.items():
        iternexts = [_ for _ in func_ir.blocks[loop.header].find_exprs('iternext')]
        if len(iternexts) != 1:
            continue
        for iternext in iternexts:
            phi = guard(get_definition, func_ir, iternext.value)
            if phi is None:
                continue
            range_call = guard(get_call_args, phi.value, range)
            if range_call is None:
                continue
            range_arg = range_call.args[0]
            len_call = guard(get_call_args, range_arg, len)
            if len_call is None:
                continue
            len_arg = len_call.args[0]
            literal_unroll_call = guard(get_definition, func_ir, len_arg)
            if literal_unroll_call is None:
                continue
            if not isinstance(literal_unroll_call, ir.Expr):
                continue
            if literal_unroll_call.op != 'call':
                continue
            literal_func = getattr(literal_unroll_call, 'func', None)
            if not literal_func:
                continue
            call_func = guard(get_definition, func_ir, literal_unroll_call.func)
            if call_func is None:
                continue
            call_func_value = call_func.value
            if call_func_value is literal_unroll:
                assert len(literal_unroll_call.args) == 1
                unroll_loops[loop] = literal_unroll_call
    return unroll_loops