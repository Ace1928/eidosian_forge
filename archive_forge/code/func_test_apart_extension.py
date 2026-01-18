from sympy.polys.partfrac import (
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (Poly, factor)
from sympy.polys.rationaltools import together
from sympy.polys.rootoftools import RootSum
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x, y, a, b, c
def test_apart_extension():
    f = 2 / (x ** 2 + 1)
    g = I / (x + I) - I / (x - I)
    assert apart(f, extension=I) == g
    assert apart(f, gaussian=True) == g
    f = x / ((x - 2) * (x + I))
    assert factor(together(apart(f)).expand()) == f
    f, g = _make_extension_example()
    from sympy.matrices import dotprodsimp
    with dotprodsimp(True):
        assert apart(f, x, extension={sqrt(2)}) == g