import common_z3 as CM_Z3
import ctypes
from .z3 import *
def myBinOp(op, *L):
    """
    >>> myAnd(*[Bool('x'),Bool('y')])
    And(x, y)

    >>> myAnd(*[Bool('x'),None])
    x

    >>> myAnd(*[Bool('x')])
    x

    >>> myAnd(*[])

    >>> myAnd(Bool('x'),Bool('y'))
    And(x, y)

    >>> myAnd(*[Bool('x'),Bool('y')])
    And(x, y)

    >>> myAnd([Bool('x'),Bool('y')])
    And(x, y)

    >>> myAnd((Bool('x'),Bool('y')))
    And(x, y)

    >>> myAnd(*[Bool('x'),Bool('y'),True])
    Traceback (most recent call last):
    ...
    AssertionError
    """
    if z3_debug():
        assert op == Z3_OP_OR or op == Z3_OP_AND or op == Z3_OP_IMPLIES
    if len(L) == 1 and (isinstance(L[0], list) or isinstance(L[0], tuple)):
        L = L[0]
    if z3_debug():
        assert all((not isinstance(val, bool) for val in L))
    L = [val for val in L if is_expr(val)]
    if L:
        if len(L) == 1:
            return L[0]
        if op == Z3_OP_OR:
            return Or(L)
        if op == Z3_OP_AND:
            return And(L)
        return Implies(L[0], L[1])
    else:
        return None