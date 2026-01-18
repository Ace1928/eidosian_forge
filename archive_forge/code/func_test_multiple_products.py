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
def test_multiple_products():
    assert product(x, (n, 1, k), (k, 1, m)) == x ** (m ** 2 / 2 + m / 2)
    assert product(f(n), (n, 1, m), (m, 1, k)) == Product(f(n), (n, 1, m), (m, 1, k)).doit()
    assert Product(f(n), (m, 1, k), (n, 1, k)).doit() == Product(Product(f(n), (m, 1, k)), (n, 1, k)).doit() == product(f(n), (m, 1, k), (n, 1, k)) == product(product(f(n), (m, 1, k)), (n, 1, k)) == Product(f(n) ** k, (n, 1, k))
    assert Product(x, (x, 1, k), (k, 1, n)).doit() == Product(factorial(k), (k, 1, n))
    assert Product(x ** k, (n, 1, k), (k, 1, m)).variables == [n, k]