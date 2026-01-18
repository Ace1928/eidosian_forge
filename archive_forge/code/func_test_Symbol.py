from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import Derivative
from sympy.core.numbers import Integer, Rational, Float, oo
from sympy.core.relational import Rel
from sympy.core.symbol import symbols
from sympy.functions import sin
from sympy.integrals.integrals import Integral
from sympy.series.order import Order
from sympy.printing.precedence import precedence, PRECEDENCE
def test_Symbol():
    assert precedence(x) == PRECEDENCE['Atom']