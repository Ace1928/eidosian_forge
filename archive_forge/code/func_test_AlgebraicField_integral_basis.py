from sympy.abc import x
from sympy.core import S
from sympy.core.numbers import AlgebraicNumber
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.numberfields.basis import round_two
from sympy.testing.pytest import raises
def test_AlgebraicField_integral_basis():
    alpha = AlgebraicNumber(sqrt(5), alias='alpha')
    k = QQ.algebraic_field(alpha)
    B0 = k.integral_basis()
    B1 = k.integral_basis(fmt='sympy')
    B2 = k.integral_basis(fmt='alg')
    assert B0 == [k([1]), k([S.Half, S.Half])]
    assert B1 == [1, S.Half + alpha / 2]
    assert B2 == [k.ext.field_element([1]), k.ext.field_element([S.Half, S.Half])]