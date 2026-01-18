from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent
def simplify_univariate(expr):
    """return a simplified version of univariate boolean expression, else ``expr``"""
    from sympy.functions.elementary.piecewise import Piecewise
    from sympy.core.relational import Eq, Ne
    if not isinstance(expr, BooleanFunction):
        return expr
    if expr.atoms(Eq, Ne):
        return expr
    c = expr
    free = c.free_symbols
    if len(free) != 1:
        return c
    x = free.pop()
    ok, i = Piecewise((0, c), evaluate=False)._intervals(x, err_on_Eq=True)
    if not ok:
        return c
    if not i:
        return false
    args = []
    for a, b, _, _ in i:
        if a is S.NegativeInfinity:
            if b is S.Infinity:
                c = true
            elif c.subs(x, b) == True:
                c = x <= b
            else:
                c = x < b
        else:
            incl_a = c.subs(x, a) == True
            incl_b = c.subs(x, b) == True
            if incl_a and incl_b:
                if b.is_infinite:
                    c = x >= a
                else:
                    c = And(a <= x, x <= b)
            elif incl_a:
                c = And(a <= x, x < b)
            elif incl_b:
                if b.is_infinite:
                    c = x > a
                else:
                    c = And(a < x, x <= b)
            else:
                c = And(a < x, x < b)
        args.append(c)
    return Or(*args)