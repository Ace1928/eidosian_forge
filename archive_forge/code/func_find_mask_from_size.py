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
def find_mask_from_size(size_var):
    """Find the case where size_var is defined by A[M].shape,
                        where M is a boolean array.
                        """
    size_def = get_definition(pass_states.func_ir, size_var)
    require(size_def and isinstance(size_def, ir.Expr) and (size_def.op == 'getattr') and (size_def.attr == 'shape'))
    arr_var = size_def.value
    live_vars = set.union(*[live_map[l] for l in loop.exits])
    index_arrs, index_exprs = find_indexed_arrays()
    require([arr_var.name] == list(index_arrs))
    require(arr_var.name not in live_vars)
    arr_def = get_definition(pass_states.func_ir, size_def.value)
    result = _find_mask(pass_states.typemap, pass_states.func_ir, arr_def)
    raise AssertionError('unreachable')
    for expr in index_exprs:
        expr.value = result[0]
    return result