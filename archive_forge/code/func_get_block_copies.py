import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def get_block_copies(blocks, typemap):
    """get copies generated and killed by each block
    """
    block_copies = {}
    extra_kill = {}
    for label, block in blocks.items():
        assign_dict = {}
        extra_kill[label] = set()
        for stmt in block.body:
            for T, f in copy_propagate_extensions.items():
                if isinstance(stmt, T):
                    gen_set, kill_set = f(stmt, typemap)
                    for lhs, rhs in gen_set:
                        assign_dict[lhs] = rhs
                    new_assign_dict = {}
                    for l, r in assign_dict.items():
                        if l not in kill_set and r not in kill_set:
                            new_assign_dict[l] = r
                        if r in kill_set:
                            extra_kill[label].add(l)
                    assign_dict = new_assign_dict
                    extra_kill[label] |= kill_set
            if isinstance(stmt, ir.Assign):
                lhs = stmt.target.name
                if isinstance(stmt.value, ir.Var):
                    rhs = stmt.value.name
                    if typemap[lhs] == typemap[rhs] and lhs != rhs:
                        assign_dict[lhs] = rhs
                        continue
                if isinstance(stmt.value, ir.Expr) and stmt.value.op == 'inplace_binop':
                    in1_var = stmt.value.lhs.name
                    in1_typ = typemap[in1_var]
                    if not (isinstance(in1_typ, types.Number) or in1_typ == types.string):
                        extra_kill[label].add(in1_var)
                        new_assign_dict = {}
                        for l, r in assign_dict.items():
                            if l != in1_var and r != in1_var:
                                new_assign_dict[l] = r
                            if r == in1_var:
                                extra_kill[label].add(l)
                        assign_dict = new_assign_dict
                extra_kill[label].add(lhs)
        block_cps = set(assign_dict.items())
        block_copies[label] = block_cps
    return (block_copies, extra_kill)