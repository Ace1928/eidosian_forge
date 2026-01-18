import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
def fix_dependencies(expr, varlist):
    """Double check if all variables in varlist are defined before
        expr is used. Try to move constant definition when the check fails.
        Bails out by raising GuardException if it can't be moved.
        """
    debug_print = _make_debug_print('fix_dependencies')
    for label, block in blocks.items():
        scope = block.scope
        body = block.body
        defined = set()
        for i in range(len(body)):
            inst = body[i]
            if isinstance(inst, ir.Assign):
                defined.add(inst.target.name)
                if inst.value is expr:
                    new_varlist = []
                    for var in varlist:
                        if var.name in defined or (var.name in livemap[label] and (not var.name in usedefs.defmap[label])):
                            debug_print(var.name, ' already defined')
                            new_varlist.append(var)
                        else:
                            debug_print(var.name, ' not yet defined')
                            var_def = get_definition(func_ir, var.name)
                            if isinstance(var_def, ir.Const):
                                loc = var.loc
                                new_var = scope.redefine('new_var', loc)
                                new_const = ir.Const(var_def.value, loc)
                                new_vardef = _new_definition(func_ir, new_var, new_const, loc)
                                new_body = []
                                new_body.extend(body[:i])
                                new_body.append(new_vardef)
                                new_body.extend(body[i:])
                                block.body = new_body
                                new_varlist.append(new_var)
                            else:
                                raise GuardException
                    return new_varlist
    raise GuardException