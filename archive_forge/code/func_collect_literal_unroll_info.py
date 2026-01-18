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
def collect_literal_unroll_info(literal_unroll_loops):
    """Finds the loops induced by `literal_unroll`, returns a list of
            unroll_info namedtuples for use in the transform pass.
            """
    literal_unroll_info = []
    for loop, literal_unroll_call in literal_unroll_loops.items():
        arg = literal_unroll_call.args[0]
        typemap = state.typemap
        resolved_arg = guard(get_definition, func_ir, arg, lhs_only=True)
        ty = typemap[resolved_arg.name]
        assert isinstance(ty, self._accepted_types)
        tuple_getitem = None
        for lbli in loop.body:
            blk = func_ir.blocks[lbli]
            for stmt in blk.body:
                if isinstance(stmt, ir.Assign):
                    if isinstance(stmt.value, ir.Expr) and stmt.value.op == 'getitem':
                        if stmt.value.value != arg:
                            dfn = guard(get_definition, func_ir, stmt.value.value)
                            if dfn is None:
                                continue
                            try:
                                args = getattr(dfn, 'args', False)
                            except KeyError:
                                continue
                            if not args:
                                continue
                            if not args[0] == arg:
                                continue
                        target_ty = state.typemap[arg.name]
                        if not isinstance(target_ty, self._accepted_types):
                            continue
                        tuple_getitem = stmt
                        break
            if tuple_getitem:
                break
        else:
            continue
        ui = unroll_info(loop, literal_unroll_call, arg, tuple_getitem)
        literal_unroll_info.append(ui)
    return literal_unroll_info