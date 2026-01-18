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
def test__dict_from_expr_if_gens():
    assert dict_from_expr(Integer(17), gens=(x,)) == ({(0,): Integer(17)}, (x,))
    assert dict_from_expr(Integer(17), gens=(x, y)) == ({(0, 0): Integer(17)}, (x, y))
    assert dict_from_expr(Integer(17), gens=(x, y, z)) == ({(0, 0, 0): Integer(17)}, (x, y, z))
    assert dict_from_expr(Integer(-17), gens=(x,)) == ({(0,): Integer(-17)}, (x,))
    assert dict_from_expr(Integer(-17), gens=(x, y)) == ({(0, 0): Integer(-17)}, (x, y))
    assert dict_from_expr(Integer(-17), gens=(x, y, z)) == ({(0, 0, 0): Integer(-17)}, (x, y, z))
    assert dict_from_expr(Integer(17) * x, gens=(x,)) == ({(1,): Integer(17)}, (x,))
    assert dict_from_expr(Integer(17) * x, gens=(x, y)) == ({(1, 0): Integer(17)}, (x, y))
    assert dict_from_expr(Integer(17) * x, gens=(x, y, z)) == ({(1, 0, 0): Integer(17)}, (x, y, z))
    assert dict_from_expr(Integer(17) * x ** 7, gens=(x,)) == ({(7,): Integer(17)}, (x,))
    assert dict_from_expr(Integer(17) * x ** 7 * y, gens=(x, y)) == ({(7, 1): Integer(17)}, (x, y))
    assert dict_from_expr(Integer(17) * x ** 7 * y * z ** 12, gens=(x, y, z)) == ({(7, 1, 12): Integer(17)}, (x, y, z))
    assert dict_from_expr(x + 2 * y + 3 * z, gens=(x,)) == ({(1,): Integer(1), (0,): 2 * y + 3 * z}, (x,))
    assert dict_from_expr(x + 2 * y + 3 * z, gens=(x, y)) == ({(1, 0): Integer(1), (0, 1): Integer(2), (0, 0): 3 * z}, (x, y))
    assert dict_from_expr(x + 2 * y + 3 * z, gens=(x, y, z)) == ({(1, 0, 0): Integer(1), (0, 1, 0): Integer(2), (0, 0, 1): Integer(3)}, (x, y, z))
    assert dict_from_expr(x * y + 2 * x * z + 3 * y * z, gens=(x,)) == ({(1,): y + 2 * z, (0,): 3 * y * z}, (x,))
    assert dict_from_expr(x * y + 2 * x * z + 3 * y * z, gens=(x, y)) == ({(1, 1): Integer(1), (1, 0): 2 * z, (0, 1): 3 * z}, (x, y))
    assert dict_from_expr(x * y + 2 * x * z + 3 * y * z, gens=(x, y, z)) == ({(1, 1, 0): Integer(1), (1, 0, 1): Integer(2), (0, 1, 1): Integer(3)}, (x, y, z))
    assert dict_from_expr(2 ** y * x, gens=(x,)) == ({(1,): 2 ** y}, (x,))
    assert dict_from_expr(Integral(x, (x, 1, 2)) + x) == ({(0, 1): 1, (1, 0): 1}, (x, Integral(x, (x, 1, 2))))
    raises(PolynomialError, lambda: dict_from_expr(2 ** y * x, gens=(x, y)))