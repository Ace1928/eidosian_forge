from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import (FallingFactorial, RisingFactorial, binomial, factorial)
from sympy.functions.special.gamma_functions import gamma
from sympy.simplify.combsimp import combsimp
from sympy.abc import x
def test_issue_6878():
    n = symbols('n', integer=True)
    assert combsimp(RisingFactorial(-10, n)) == 3628800 * (-1) ** n / factorial(10 - n)