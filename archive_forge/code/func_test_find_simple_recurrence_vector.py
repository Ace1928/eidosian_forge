from sympy.concrete.guess import (
from sympy.concrete.products import Product
from sympy.core.function import Function
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.elementary.exponential import exp
def test_find_simple_recurrence_vector():
    assert find_simple_recurrence_vector([fibonacci(k) for k in range(12)]) == [1, -1, -1]