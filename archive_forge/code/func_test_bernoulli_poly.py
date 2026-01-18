from sympy.core.numbers import Rational as Q
from sympy.polys.polytools import Poly
from sympy.testing.pytest import raises
from sympy.polys.appellseqs import (bernoulli_poly, bernoulli_c_poly,
from sympy.abc import x
def test_bernoulli_poly():
    raises(ValueError, lambda: bernoulli_poly(-1, x))
    assert bernoulli_poly(1, x, polys=True) == Poly(x - Q(1, 2))
    assert bernoulli_poly(0, x) == 1
    assert bernoulli_poly(1, x) == x - Q(1, 2)
    assert bernoulli_poly(2, x) == x ** 2 - x + Q(1, 6)
    assert bernoulli_poly(3, x) == x ** 3 - Q(3, 2) * x ** 2 + Q(1, 2) * x
    assert bernoulli_poly(4, x) == x ** 4 - 2 * x ** 3 + x ** 2 - Q(1, 30)
    assert bernoulli_poly(5, x) == x ** 5 - Q(5, 2) * x ** 4 + Q(5, 3) * x ** 3 - Q(1, 6) * x
    assert bernoulli_poly(6, x) == x ** 6 - 3 * x ** 5 + Q(5, 2) * x ** 4 - Q(1, 2) * x ** 2 + Q(1, 42)
    assert bernoulli_poly(1).dummy_eq(x - Q(1, 2))
    assert bernoulli_poly(1, polys=True) == Poly(x - Q(1, 2))