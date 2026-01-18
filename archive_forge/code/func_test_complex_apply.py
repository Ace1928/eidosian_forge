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
def test_complex_apply():
    n, m = symbols('n,m')
    o = Bd(0) * B(0) * Bd(1) * B(0)
    e = apply_operators(o * BKet([n, m]))
    answer = sqrt(n) * sqrt(m + 1) * (-1 + n) * BKet([-1 + n, 1 + m])
    assert expand(e) == expand(answer)