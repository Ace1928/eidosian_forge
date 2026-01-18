from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
def optimise_numeric_binop(operator, node, ret_type, arg0, arg1):
    """
    Optimise math operators for (likely) float or small integer operations.
    """
    num_nodes = (ExprNodes.IntNode, ExprNodes.FloatNode)
    if isinstance(arg1, num_nodes):
        if arg0.type is not PyrexTypes.py_object_type:
            return None
        numval = arg1
        arg_order = 'ObjC'
    elif isinstance(arg0, num_nodes):
        if arg1.type is not PyrexTypes.py_object_type:
            return None
        numval = arg0
        arg_order = 'CObj'
    else:
        return None
    if not numval.has_constant_result():
        return None
    is_float = isinstance(numval, ExprNodes.FloatNode)
    num_type = PyrexTypes.c_double_type if is_float else PyrexTypes.c_long_type
    if is_float:
        if operator not in ('Add', 'Subtract', 'Remainder', 'TrueDivide', 'Divide', 'Eq', 'Ne'):
            return None
    elif operator == 'Divide':
        return None
    elif abs(numval.constant_result) > 2 ** 30:
        return None
    if operator in ('TrueDivide', 'FloorDivide', 'Divide', 'Remainder'):
        if arg1.constant_result == 0:
            return None
    extra_args = []
    extra_args.append((ExprNodes.FloatNode if is_float else ExprNodes.IntNode)(numval.pos, value=numval.value, constant_result=numval.constant_result, type=num_type))
    inplace = node.inplace if isinstance(node, ExprNodes.NumBinopNode) else False
    extra_args.append(ExprNodes.BoolNode(node.pos, value=inplace, constant_result=inplace))
    if is_float or operator not in ('Eq', 'Ne'):
        zerodivision_check = arg_order == 'CObj' and (not node.cdivision if isinstance(node, ExprNodes.DivNode) else False)
        extra_args.append(ExprNodes.BoolNode(node.pos, value=zerodivision_check, constant_result=zerodivision_check))
    utility_code = TempitaUtilityCode.load_cached('PyFloatBinop' if is_float else 'PyIntCompare' if operator in ('Eq', 'Ne') else 'PyIntBinop', 'Optimize.c', context=dict(op=operator, order=arg_order, ret_type=ret_type))
    func_cname = '__Pyx_Py%s_%s%s%s' % ('Float' if is_float else 'Int', '' if ret_type.is_pyobject else 'Bool', operator, arg_order)
    return (func_cname, utility_code, extra_args, num_type)