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
def apply_copies_parfor(parfor, var_dict, name_var_table, typemap, calltypes, save_copies):
    """apply copy propagate recursively in parfor"""
    for i, pattern in enumerate(parfor.patterns):
        if pattern[0] == 'stencil':
            parfor.patterns[i] = ('stencil', replace_vars_inner(pattern[1], var_dict))
    for l in parfor.loop_nests:
        l.start = replace_vars_inner(l.start, var_dict)
        l.stop = replace_vars_inner(l.stop, var_dict)
        l.step = replace_vars_inner(l.step, var_dict)
    blocks = wrap_parfor_blocks(parfor)
    assign_list = []
    for lhs_name, rhs in var_dict.items():
        assign_list.append(ir.Assign(rhs, name_var_table[lhs_name], ir.Loc('dummy', -1)))
    blocks[0].body = assign_list + blocks[0].body
    in_copies_parfor, out_copies_parfor = copy_propagate(blocks, typemap)
    apply_copy_propagate(blocks, in_copies_parfor, name_var_table, typemap, calltypes, save_copies)
    unwrap_parfor_blocks(parfor)
    blocks[0].body = blocks[0].body[len(assign_list):]
    return