from sympy.parsing.maxima import parse_maxima
from sympy.core.numbers import (E, Rational, oo)
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.abc import x
def test_injection():
    parse_maxima('c: x+1', globals=globals())
    assert c == x + 1
    parse_maxima('g: sqrt(81)', globals=globals())
    assert g == 9