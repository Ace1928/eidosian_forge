import operator
from functools import reduce
from collections import namedtuple, defaultdict
from .controlflow import CFGraph
from numba.core import types, errors, ir, consts
from numba.misc import special
def compute_use_defs(blocks):
    """
    Find variable use/def per block.
    """
    var_use_map = {}
    var_def_map = {}
    for offset, ir_block in blocks.items():
        var_use_map[offset] = use_set = set()
        var_def_map[offset] = def_set = set()
        for stmt in ir_block.body:
            if type(stmt) in ir_extension_usedefs:
                func = ir_extension_usedefs[type(stmt)]
                func(stmt, use_set, def_set)
                continue
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Inst):
                    rhs_set = set((var.name for var in stmt.value.list_vars()))
                elif isinstance(stmt.value, ir.Var):
                    rhs_set = set([stmt.value.name])
                elif isinstance(stmt.value, (ir.Arg, ir.Const, ir.Global, ir.FreeVar)):
                    rhs_set = ()
                else:
                    raise AssertionError('unreachable', type(stmt.value))
                if stmt.target.name not in rhs_set:
                    def_set.add(stmt.target.name)
            for var in stmt.list_vars():
                if var.name not in def_set:
                    use_set.add(var.name)
    return _use_defs_result(usemap=var_use_map, defmap=var_def_map)