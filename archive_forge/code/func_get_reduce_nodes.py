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
def get_reduce_nodes(reduction_node, nodes, func_ir):
    """
    Get nodes that combine the reduction variable with a sentinel variable.
    Recognizes the first node that combines the reduction variable with another
    variable.
    """
    reduce_nodes = None
    defs = {}

    def cyclic_lookup(var, varonly=True, start=None):
        """Lookup definition of ``var``.
        Returns ``None`` if variable definition forms a cycle.
        """
        lookedup_var = defs.get(var.name, None)
        if isinstance(lookedup_var, ir.Var):
            if start is None:
                start = lookedup_var
            elif start == lookedup_var:
                return None
            return cyclic_lookup(lookedup_var, start=start)
        else:
            return var if varonly or lookedup_var is None else lookedup_var

    def noncyclic_lookup(*args, **kwargs):
        """Similar to cyclic_lookup but raise AssertionError if a cycle is
        detected.
        """
        res = cyclic_lookup(*args, **kwargs)
        if res is None:
            raise AssertionError('unexpected cycle in lookup()')
        return res
    name = reduction_node.name
    unversioned_name = reduction_node.unversioned_name
    for i, stmt in enumerate(nodes):
        lhs = stmt.target
        rhs = stmt.value
        defs[lhs.name] = rhs
        if isinstance(rhs, ir.Var) and rhs.name in defs:
            rhs = cyclic_lookup(rhs)
        if isinstance(rhs, ir.Expr):
            in_vars = set((noncyclic_lookup(v, True).name for v in rhs.list_vars()))
            if name in in_vars:
                if i + 1 < len(nodes) and (not isinstance(nodes[i + 1], ir.Assign) or nodes[i + 1].target.unversioned_name != unversioned_name):
                    foundj = None
                    for j, jstmt in enumerate(nodes[i + 1:]):
                        if isinstance(jstmt, ir.Assign) and jstmt.value == lhs:
                            foundj = i + j + 1
                            break
                    if foundj is not None:
                        nodes = nodes[:i + 1] + nodes[foundj:foundj + 1] + nodes[i + 1:foundj] + nodes[foundj + 1:]
                if not (i + 1 < len(nodes) and isinstance(nodes[i + 1], ir.Assign) and (nodes[i + 1].target.unversioned_name == unversioned_name)) and lhs.unversioned_name != unversioned_name:
                    raise ValueError(f'Use of reduction variable {unversioned_name!r} other than in a supported reduction function is not permitted.')
                if not supported_reduction(rhs, func_ir):
                    raise ValueError('Use of reduction variable ' + unversioned_name + ' in an unsupported reduction function.')
                args = [(x.name, noncyclic_lookup(x, True)) for x in get_expr_args(rhs)]
                non_red_args = [x for x, y in args if y.name != name]
                assert len(non_red_args) == 1
                args = [(x, y) for x, y in args if x != y.name]
                replace_dict = dict(args)
                replace_dict[non_red_args[0]] = ir.Var(lhs.scope, name + '#init', lhs.loc)
                replace_vars_inner(rhs, replace_dict)
                reduce_nodes = nodes[i:]
                break
    return reduce_nodes