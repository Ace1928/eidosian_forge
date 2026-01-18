import common_z3 as CM_Z3
import ctypes
from .z3 import *
def is_expr_var(v):
    """
    EXAMPLES:

    >>> is_expr_var(Int('7'))
    True
    >>> is_expr_var(IntVal('7'))
    False
    >>> is_expr_var(Bool('y'))
    True
    >>> is_expr_var(Int('x') + 7 == Int('y'))
    False
    >>> LOnOff, (On,Off) = EnumSort("LOnOff",['On','Off'])
    >>> Block,Reset,SafetyInjection=Consts("Block Reset SafetyInjection",LOnOff)
    >>> is_expr_var(LOnOff)
    False
    >>> is_expr_var(On)
    False
    >>> is_expr_var(Block)
    True
    >>> is_expr_var(SafetyInjection)
    True
    """
    return is_const(v) and v.decl().kind() == Z3_OP_UNINTERPRETED