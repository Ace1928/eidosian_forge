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
def remove_dead_parfor_recursive(parfor, lives, arg_aliases, alias_map, func_ir, typemap):
    """create a dummy function from parfor and call remove dead recursively
    """
    blocks = parfor.loop_body.copy()
    first_body_block = min(blocks.keys())
    assert first_body_block > 0
    last_label = max(blocks.keys())
    "\n      Previously, this statement used lives_n_aliases.  That had the effect of\n      keeping variables in the init_block alive if they aliased an array that\n      was later written to.  By using just lives to indicate which variables\n      names are live at exit of the parfor but then using alias_map for the\n      actual recursive dead code removal, we keep any writes to aliased arrays\n      alive but also allow aliasing assignments (i.e., a = b) to be eliminated\n      so long as 'b' is not written to through the variable 'a' later on.\n      This makes assignment handling of remove_dead_block work properly since\n      it allows distinguishing between live variables and their aliases.\n    "
    return_label, tuple_var = _add_liveness_return_block(blocks, lives, typemap)
    scope = blocks[last_label].scope
    branchcond = ir.Var(scope, mk_unique_var('$branchcond'), ir.Loc('parfors_dummy', -1))
    typemap[branchcond.name] = types.boolean
    branch = ir.Branch(branchcond, first_body_block, return_label, ir.Loc('parfors_dummy', -1))
    blocks[last_label].body.append(branch)
    blocks[0] = parfor.init_block
    blocks[0].body.append(ir.Jump(first_body_block, ir.Loc('parfors_dummy', -1)))
    remove_dead(blocks, arg_aliases, func_ir, typemap, alias_map, arg_aliases)
    typemap.pop(tuple_var.name)
    blocks[0].body.pop()
    blocks[last_label].body.pop()
    return