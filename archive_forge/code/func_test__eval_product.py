from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import Sum
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (rf, factorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.simplify.combsimp import combsimp
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import raises
def test__eval_product():
    from sympy.abc import i, n
    a = Function('a')
    assert product(2 * a(i), (i, 1, n)) == 2 ** n * Product(a(i), (i, 1, n))
    assert product(2 ** i, (i, 1, n)) == 2 ** (n * (n + 1) / 2)
    k, m = symbols('k m', integer=True)
    assert product(2 ** i, (i, k, m)) == 2 ** (-k ** 2 / 2 + k / 2 + m ** 2 / 2 + m / 2)
    n = Symbol('n', negative=True, integer=True)
    p = Symbol('p', positive=True, integer=True)
    assert product(2 ** i, (i, n, p)) == 2 ** (-n ** 2 / 2 + n / 2 + p ** 2 / 2 + p / 2)
    assert product(2 ** i, (i, p, n)) == 2 ** (n ** 2 / 2 + n / 2 - p ** 2 / 2 + p / 2)