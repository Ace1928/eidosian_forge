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
def add_indices_to_kernel(self, kernel, index_names, ndim, neighborhood, standard_indexed, typemap, calltypes):
    """
        Transforms the stencil kernel as specified by the user into one
        that includes each dimension's index variable as part of the getitem
        calls.  So, in effect array[-1] becomes array[index0-1].
        """
    const_dict = {}
    kernel_consts = []
    if config.DEBUG_ARRAY_OPT >= 1:
        print('add_indices_to_kernel', ndim, neighborhood)
        ir_utils.dump_blocks(kernel.blocks)
    if neighborhood is None:
        need_to_calc_kernel = True
    else:
        need_to_calc_kernel = False
        if len(neighborhood) != ndim:
            raise ValueError('%d dimensional neighborhood specified for %d dimensional input array' % (len(neighborhood), ndim))
    tuple_table = ir_utils.get_tuple_table(kernel.blocks)
    relatively_indexed = set()
    for block in kernel.blocks.values():
        scope = block.scope
        loc = block.loc
        new_body = []
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Const):
                if config.DEBUG_ARRAY_OPT >= 1:
                    print('remembering in const_dict', stmt.target.name, stmt.value.value)
                const_dict[stmt.target.name] = stmt.value.value
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op in ['setitem', 'static_setitem']) and (stmt.value.value.name in kernel.arg_names) or (isinstance(stmt, ir.SetItem) and stmt.target.name in kernel.arg_names):
                raise ValueError('Assignments to arrays passed to stencil kernels is not allowed.')
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op in ['getitem', 'static_getitem']) and (stmt.value.value.name in kernel.arg_names) and (stmt.value.value.name not in standard_indexed):
                if stmt.value.op == 'getitem':
                    stmt_index_var = stmt.value.index
                else:
                    stmt_index_var = stmt.value.index_var
                relatively_indexed.add(stmt.value.value.name)
                if need_to_calc_kernel:
                    assert hasattr(stmt_index_var, 'name')
                    if stmt_index_var.name in tuple_table:
                        kernel_consts += [tuple_table[stmt_index_var.name]]
                    elif stmt_index_var.name in const_dict:
                        kernel_consts += [const_dict[stmt_index_var.name]]
                    else:
                        raise NumbaValueError("stencil kernel index is not constant, 'neighborhood' option required")
                if ndim == 1:
                    index_var = ir.Var(scope, index_names[0], loc)
                    tmpvar = scope.redefine('stencil_index', loc)
                    stmt_index_var_typ = typemap[stmt_index_var.name]
                    if isinstance(stmt_index_var_typ, types.misc.SliceType):
                        sa_var = scope.redefine('slice_addition', loc)
                        sa_func = numba.njit(slice_addition)
                        sa_func_typ = types.functions.Dispatcher(sa_func)
                        typemap[sa_var.name] = sa_func_typ
                        g_sa = ir.Global('slice_addition', sa_func, loc)
                        new_body.append(ir.Assign(g_sa, sa_var, loc))
                        slice_addition_call = ir.Expr.call(sa_var, [stmt_index_var, index_var], (), loc)
                        calltypes[slice_addition_call] = sa_func_typ.get_call_type(self._typingctx, [stmt_index_var_typ, types.intp], {})
                        new_body.append(ir.Assign(slice_addition_call, tmpvar, loc))
                        new_body.append(ir.Assign(ir.Expr.getitem(stmt.value.value, tmpvar, loc), stmt.target, loc))
                    else:
                        acc_call = ir.Expr.binop(operator.add, stmt_index_var, index_var, loc)
                        new_body.append(ir.Assign(acc_call, tmpvar, loc))
                        new_body.append(ir.Assign(ir.Expr.getitem(stmt.value.value, tmpvar, loc), stmt.target, loc))
                else:
                    index_vars = []
                    sum_results = []
                    s_index_var = scope.redefine('stencil_index', loc)
                    const_index_vars = []
                    ind_stencils = []
                    stmt_index_var_typ = typemap[stmt_index_var.name]
                    for dim in range(ndim):
                        tmpvar = scope.redefine('const_index', loc)
                        new_body.append(ir.Assign(ir.Const(dim, loc), tmpvar, loc))
                        const_index_vars += [tmpvar]
                        index_var = ir.Var(scope, index_names[dim], loc)
                        index_vars += [index_var]
                        tmpvar = scope.redefine('ind_stencil_index', loc)
                        ind_stencils += [tmpvar]
                        getitemvar = scope.redefine('getitem', loc)
                        getitemcall = ir.Expr.getitem(stmt_index_var, const_index_vars[dim], loc)
                        new_body.append(ir.Assign(getitemcall, getitemvar, loc))
                        if isinstance(stmt_index_var_typ, types.ConstSized):
                            one_index_typ = stmt_index_var_typ[dim]
                        else:
                            one_index_typ = stmt_index_var_typ[:]
                        if isinstance(one_index_typ, types.misc.SliceType):
                            sa_var = scope.redefine('slice_addition', loc)
                            sa_func = numba.njit(slice_addition)
                            sa_func_typ = types.functions.Dispatcher(sa_func)
                            typemap[sa_var.name] = sa_func_typ
                            g_sa = ir.Global('slice_addition', sa_func, loc)
                            new_body.append(ir.Assign(g_sa, sa_var, loc))
                            slice_addition_call = ir.Expr.call(sa_var, [getitemvar, index_vars[dim]], (), loc)
                            calltypes[slice_addition_call] = sa_func_typ.get_call_type(self._typingctx, [one_index_typ, types.intp], {})
                            new_body.append(ir.Assign(slice_addition_call, tmpvar, loc))
                        else:
                            acc_call = ir.Expr.binop(operator.add, getitemvar, index_vars[dim], loc)
                            new_body.append(ir.Assign(acc_call, tmpvar, loc))
                    tuple_call = ir.Expr.build_tuple(ind_stencils, loc)
                    new_body.append(ir.Assign(tuple_call, s_index_var, loc))
                    new_body.append(ir.Assign(ir.Expr.getitem(stmt.value.value, s_index_var, loc), stmt.target, loc))
            else:
                new_body.append(stmt)
        block.body = new_body
    if need_to_calc_kernel:
        neighborhood = [[0, 0] for _ in range(ndim)]
        if len(kernel_consts) == 0:
            raise NumbaValueError('Stencil kernel with no accesses to relatively indexed arrays.')
        for index in kernel_consts:
            if isinstance(index, tuple) or isinstance(index, list):
                for i in range(len(index)):
                    te = index[i]
                    if isinstance(te, ir.Var) and te.name in const_dict:
                        te = const_dict[te.name]
                    if isinstance(te, int):
                        neighborhood[i][0] = min(neighborhood[i][0], te)
                        neighborhood[i][1] = max(neighborhood[i][1], te)
                    else:
                        raise NumbaValueError("stencil kernel index is not constant,'neighborhood' option required")
                index_len = len(index)
            elif isinstance(index, int):
                neighborhood[0][0] = min(neighborhood[0][0], index)
                neighborhood[0][1] = max(neighborhood[0][1], index)
                index_len = 1
            else:
                raise NumbaValueError('Non-tuple or non-integer used as stencil index.')
            if index_len != ndim:
                raise NumbaValueError('Stencil index does not match array dimensionality.')
    return (neighborhood, relatively_indexed)