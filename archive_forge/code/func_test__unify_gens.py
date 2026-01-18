from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.testing.pytest import raises
from sympy.polys.polyutils import (
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.domains import ZZ
def test__unify_gens():
    assert _unify_gens([], []) == ()
    assert _unify_gens([x], [x]) == (x,)
    assert _unify_gens([y], [y]) == (y,)
    assert _unify_gens([x, y], [x]) == (x, y)
    assert _unify_gens([x], [x, y]) == (x, y)
    assert _unify_gens([x, y], [x, y]) == (x, y)
    assert _unify_gens([y, x], [y, x]) == (y, x)
    assert _unify_gens([x], [y]) == (x, y)
    assert _unify_gens([y], [x]) == (y, x)
    assert _unify_gens([x], [y, x]) == (y, x)
    assert _unify_gens([y, x], [x]) == (y, x)
    assert _unify_gens([x, y, z], [x, y, z]) == (x, y, z)
    assert _unify_gens([z, y, x], [x, y, z]) == (z, y, x)
    assert _unify_gens([x, y, z], [z, y, x]) == (x, y, z)
    assert _unify_gens([z, y, x], [z, y, x]) == (z, y, x)
    assert _unify_gens([x, y, z], [t, x, p, q, z]) == (t, x, y, p, q, z)