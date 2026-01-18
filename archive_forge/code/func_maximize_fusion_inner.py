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
def maximize_fusion_inner(func_ir, block, call_table, alias_map, arg_aliases, up_direction=True):
    order_changed = False
    i = 0
    while i < len(block.body) - 2:
        stmt = block.body[i]
        next_stmt = block.body[i + 1]
        can_reorder = _can_reorder_stmts(stmt, next_stmt, func_ir, call_table, alias_map, arg_aliases) if up_direction else _can_reorder_stmts(next_stmt, stmt, func_ir, call_table, alias_map, arg_aliases)
        if can_reorder:
            block.body[i] = next_stmt
            block.body[i + 1] = stmt
            order_changed = True
        i += 1
    return order_changed