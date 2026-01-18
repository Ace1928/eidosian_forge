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
def remove_dead_parfor(parfor, lives, lives_n_aliases, arg_aliases, alias_map, func_ir, typemap):
    """ remove dead code inside parfor including get/sets
    """
    with dummy_return_in_loop_body(parfor.loop_body):
        labels = find_topo_order(parfor.loop_body)
    first_label = labels[0]
    first_block_saved_values = {}
    _update_parfor_get_setitems(parfor.loop_body[first_label].body, parfor.index_var, alias_map, first_block_saved_values, lives_n_aliases)
    saved_arrs = set(first_block_saved_values.keys())
    for l in labels:
        if l == first_label:
            continue
        for stmt in parfor.loop_body[l].body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op == 'getitem') and (stmt.value.index.name == parfor.index_var.name):
                continue
            varnames = set((v.name for v in stmt.list_vars()))
            rm_arrs = varnames & saved_arrs
            for a in rm_arrs:
                first_block_saved_values.pop(a, None)
    for l in labels:
        if l == first_label:
            continue
        block = parfor.loop_body[l]
        saved_values = first_block_saved_values.copy()
        _update_parfor_get_setitems(block.body, parfor.index_var, alias_map, saved_values, lives_n_aliases)
    blocks = parfor.loop_body.copy()
    last_label = max(blocks.keys())
    return_label, tuple_var = _add_liveness_return_block(blocks, lives_n_aliases, typemap)
    jump = ir.Jump(return_label, ir.Loc('parfors_dummy', -1))
    blocks[last_label].body.append(jump)
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
    alias_set = set(alias_map.keys())
    for label, block in blocks.items():
        new_body = []
        in_lives = {v.name for v in block.terminator.list_vars()}
        for out_blk, _data in cfg.successors(label):
            in_lives |= live_map[out_blk]
        for stmt in reversed(block.body):
            alias_lives = in_lives & alias_set
            for v in alias_lives:
                in_lives |= alias_map[v]
            if isinstance(stmt, (ir.StaticSetItem, ir.SetItem)) and get_index_var(stmt).name == parfor.index_var.name and (stmt.target.name not in in_lives) and (stmt.target.name not in arg_aliases):
                continue
            in_lives |= {v.name for v in stmt.list_vars()}
            new_body.append(stmt)
        new_body.reverse()
        block.body = new_body
    typemap.pop(tuple_var.name)
    blocks[last_label].body.pop()
    '\n      Process parfor body recursively.\n      Note that this is the only place in this function that uses the\n      argument lives instead of lives_n_aliases.  The former does not\n      include the aliases of live variables but only the live variable\n      names themselves.  See a comment in this function for how that\n      is used.\n    '
    remove_dead_parfor_recursive(parfor, lives, arg_aliases, alias_map, func_ir, typemap)
    is_empty = len(parfor.init_block.body) == 0
    for block in parfor.loop_body.values():
        is_empty &= len(block.body) == 0
    if is_empty:
        return None
    return parfor