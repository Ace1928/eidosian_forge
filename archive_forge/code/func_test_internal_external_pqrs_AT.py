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
def test_internal_external_pqrs_AT():
    ii, jj = symbols('i j')
    aa, bb = symbols('a b')
    k, l = symbols('k l', cls=Dummy)
    c, d = symbols('c d', cls=Dummy)
    exprs = [atv(k, l, c, d) * att(aa, c, ii, k) * att(bb, d, jj, l), atv(l, k, c, d) * att(aa, c, ii, l) * att(bb, d, jj, k), atv(k, l, d, c) * att(aa, d, ii, k) * att(bb, c, jj, l), atv(l, k, d, c) * att(aa, d, ii, l) * att(bb, c, jj, k)]
    for permut in exprs[1:]:
        assert substitute_dummies(exprs[0]) == substitute_dummies(permut)