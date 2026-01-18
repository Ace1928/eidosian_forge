import types as pytypes  # avoid confusion with numba.types
import sys, math
import os
import textwrap
import copy
import inspect
import linecache
from functools import reduce
from collections import defaultdict, OrderedDict, namedtuple
from contextlib import contextmanager
import operator
from dataclasses import make_dataclass
import warnings
from llvmlite import ir as lir
from numba.core.imputils import impl_ret_untracked
import numba.core.ir
from numba.core import types, typing, utils, errors, ir, analysis, postproc, rewrites, typeinfer, config, ir_utils
from numba import prange, pndindex
from numba.np.npdatetime_helpers import datetime_minimum, datetime_maximum
from numba.np.numpy_support import as_dtype, numpy_version
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.stencils.stencilparfor import StencilPass
from numba.core.extending import register_jitable, lower_builtin
from numba.core.ir_utils import (
from numba.core.analysis import (compute_use_defs, compute_live_map,
from numba.core.controlflow import CFGraph
from numba.core.typing import npydecl, signature
from numba.core.types.functions import Function
from numba.parfors.array_analysis import (random_int_args, random_1arg_size,
from numba.core.extending import overload
import copy
import numpy
import numpy as np
from numba.parfors import array_analysis
import numba.cpython.builtins
from numba.stencils import stencilparfor
def replace_func():
    func_def = get_definition(self.func_ir, expr.func)
    callname = find_callname(self.func_ir, expr)
    repl_func = self.replace_functions_map.get(callname, None)
    if repl_func is None and len(callname) == 2 and isinstance(callname[1], ir.Var) and isinstance(self.typemap[callname[1].name], types.npytypes.Array):
        repl_func = replace_functions_ndarray.get(callname[0], None)
        if repl_func is not None:
            expr.args.insert(0, callname[1])
    require(repl_func is not None)
    typs = tuple((self.typemap[x.name] for x in expr.args))
    kws_typs = {k: self.typemap[x.name] for k, x in expr.kws}
    try:
        new_func = repl_func(lhs_typ, *typs, **kws_typs)
    except:
        new_func = None
    require(new_func is not None)
    typs = utils.pysignature(new_func).bind(*typs, **kws_typs).args
    g = copy.copy(self.func_ir.func_id.func.__globals__)
    g['numba'] = numba
    g['np'] = numpy
    g['math'] = math
    check = replace_functions_checkers_map.get(callname, None)
    if check is not None:
        g[check.name] = check.func
    new_blocks, _ = inline_closure_call(self.func_ir, g, block, i, new_func, self.typingctx, self.targetctx, typs, self.typemap, self.calltypes, work_list)
    call_table = get_call_table(new_blocks, topological_ordering=False)
    for call in call_table:
        for k, v in call.items():
            if v[0] == 'internal_prange':
                swapped[k] = [callname, repl_func.__name__, func_def, block.body[i].loc]
                break
    return True