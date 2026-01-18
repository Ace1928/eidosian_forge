from sympy.core import Function, S, Mul, Pow, Add
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.function import expand_func
from sympy.core.symbol import Dummy
from sympy.functions import gamma, sqrt, sin
from sympy.polys import factor, cancel
from sympy.utilities.iterables import sift, uniq
def compute_ST(expr):
    if expr in inv:
        return inv[expr]
    return (expr.free_symbols, expr.atoms(Function).union({e.exp for e in expr.atoms(Pow)}))