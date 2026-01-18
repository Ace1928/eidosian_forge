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
def get_call_args(init_arg, want):
    some_call = get_definition(func_ir, init_arg)
    if not isinstance(some_call, ir.Expr):
        raise GuardException
    if not some_call.op == 'call':
        raise GuardException
    the_global = get_definition(func_ir, some_call.func)
    if not isinstance(the_global, ir.Global):
        raise GuardException
    if the_global.value is not want:
        raise GuardException
    return some_call