from . import (
from .Errors import error
from . import PyrexTypes
from .UtilityCode import CythonUtilityCode
from .Code import TempitaUtilityCode, UtilityCode
from .Visitor import PrintTree, TreeVisitor, VisitorTransform
def generate_ufunc_initialization(converters, cfunc_nodes, original_node):
    global_scope = converters[0].global_scope
    ufunc_funcs_name = global_scope.next_id(Naming.pyrex_prefix + 'funcs')
    ufunc_types_name = global_scope.next_id(Naming.pyrex_prefix + 'types')
    ufunc_data_name = global_scope.next_id(Naming.pyrex_prefix + 'data')
    type_constants = []
    narg_in = None
    narg_out = None
    for c in converters:
        in_const = [d.type_constant for d in c.in_definitions]
        if narg_in is not None:
            assert narg_in == len(in_const)
        else:
            narg_in = len(in_const)
        type_constants.extend(in_const)
        out_const = [d.type_constant for d in c.out_definitions]
        if narg_out is not None:
            assert narg_out == len(out_const)
        else:
            narg_out = len(out_const)
        type_constants.extend(out_const)
    func_cnames = [cfnode.entry.cname for cfnode in cfunc_nodes]
    context = dict(ufunc_funcs_name=ufunc_funcs_name, func_cnames=func_cnames, ufunc_types_name=ufunc_types_name, type_constants=type_constants, ufunc_data_name=ufunc_data_name)
    global_scope.use_utility_code(TempitaUtilityCode.load('UFuncConsts', 'UFuncs_C.c', context=context))
    pos = original_node.pos
    func_name = original_node.entry.name
    docstr = original_node.doc
    args_to_func = '%s(), %s, %s(), %s, %s, %s, PyUFunc_None, "%s", %s, 0' % (ufunc_funcs_name, ufunc_data_name, ufunc_types_name, len(func_cnames), narg_in, narg_out, func_name, docstr.as_c_string_literal() if docstr else 'NULL')
    call_node = ExprNodes.PythonCapiCallNode(pos, function_name='PyUFunc_FromFuncAndData', func_type=PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('dummy', PyrexTypes.c_void_ptr_type, None)]), args=[ExprNodes.ConstNode(pos, type=PyrexTypes.c_void_ptr_type, value=args_to_func)])
    lhs_entry = global_scope.declare_var(func_name, PyrexTypes.py_object_type, pos)
    assgn_node = Nodes.SingleAssignmentNode(pos, lhs=ExprNodes.NameNode(pos, name=func_name, type=PyrexTypes.py_object_type, entry=lhs_entry), rhs=call_node)
    return assgn_node