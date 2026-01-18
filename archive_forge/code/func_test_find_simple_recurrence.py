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
def test_find_simple_recurrence():
    a = Function('a')
    n = Symbol('n')
    assert find_simple_recurrence([fibonacci(k) for k in range(12)]) == -a(n) - a(n + 1) + a(n + 2)
    f = Function('a')
    i = Symbol('n')
    a = [1, 1, 1]
    for k in range(15):
        a.append(5 * a[-1] - 3 * a[-2] + 8 * a[-3])
    assert find_simple_recurrence(a, A=f, N=i) == -8 * f(i) + 3 * f(i + 1) - 5 * f(i + 2) + f(i + 3)
    assert find_simple_recurrence([0, 2, 15, 74, 12, 3, 0, 1, 2, 85, 4, 5, 63]) == 0