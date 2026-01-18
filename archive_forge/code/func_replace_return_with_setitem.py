import copy
import numpy as np
from llvmlite import ir as lir
from numba.core import types, typing, utils, ir, config, ir_utils, registry
from numba.core.typing.templates import (CallableTemplate, signature,
from numba.core.imputils import lower_builtin
from numba.core.extending import register_jitable
from numba.core.errors import NumbaValueError
from numba.misc.special import literal_unroll
import numba
import operator
from numba.np import numpy_support
def replace_return_with_setitem(self, blocks, index_vars, out_name):
    """
        Find return statements in the IR and replace them with a SetItem
        call of the value "returned" by the kernel into the result array.
        Returns the block labels that contained return statements.
        """
    ret_blocks = []
    for label, block in blocks.items():
        scope = block.scope
        loc = block.loc
        new_body = []
        for stmt in block.body:
            if isinstance(stmt, ir.Return):
                ret_blocks.append(label)
                if len(index_vars) == 1:
                    rvar = ir.Var(scope, out_name, loc)
                    ivar = ir.Var(scope, index_vars[0], loc)
                    new_body.append(ir.SetItem(rvar, ivar, stmt.value, loc))
                else:
                    var_index_vars = []
                    for one_var in index_vars:
                        index_var = ir.Var(scope, one_var, loc)
                        var_index_vars += [index_var]
                    s_index_var = scope.redefine('stencil_index', loc)
                    tuple_call = ir.Expr.build_tuple(var_index_vars, loc)
                    new_body.append(ir.Assign(tuple_call, s_index_var, loc))
                    rvar = ir.Var(scope, out_name, loc)
                    si = ir.SetItem(rvar, s_index_var, stmt.value, loc)
                    new_body.append(si)
            else:
                new_body.append(stmt)
        block.body = new_body
    return ret_blocks