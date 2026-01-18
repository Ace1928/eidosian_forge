from sympy.physics.secondquant import (
from sympy.concrete.summations import Sum
from sympy.core.function import (Function, expand)
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.repr import srepr
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import slow, raises
from sympy.printing.latex import latex
def test_dummy_order_inner_outer_lines_VT1T1T1():
    ii = symbols('i', below_fermi=True)
    aa = symbols('a', above_fermi=True)
    k, l = symbols('k l', below_fermi=True, cls=Dummy)
    c, d = symbols('c d', above_fermi=True, cls=Dummy)
    v = Function('v')
    t = Function('t')
    dums = _get_ordered_dummies
    exprs = [v(k, l, c, d) * t(c, ii) * t(d, l) * t(aa, k), v(l, k, c, d) * t(c, ii) * t(d, k) * t(aa, l), v(k, l, d, c) * t(d, ii) * t(c, l) * t(aa, k), v(l, k, d, c) * t(d, ii) * t(c, k) * t(aa, l)]
    for permut in exprs[1:]:
        assert dums(exprs[0]) != dums(permut)
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)