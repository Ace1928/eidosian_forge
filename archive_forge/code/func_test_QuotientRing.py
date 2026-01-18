from sympy.polys.domains.integerring import ZZ
from sympy.polys.domains.rationalfield import QQ
from sympy.abc import x, y
from sympy.polys.polyerrors import NotReversible
from sympy.testing.pytest import raises
def test_QuotientRing():
    I = QQ.old_poly_ring(x).ideal(x ** 2 + 1)
    R = QQ.old_poly_ring(x) / I
    assert R == QQ.old_poly_ring(x) / [x ** 2 + 1]
    assert R == QQ.old_poly_ring(x) / QQ.old_poly_ring(x).ideal(x ** 2 + 1)
    assert R != QQ.old_poly_ring(x)
    assert R.convert(1) / x == -x + I
    assert -1 + I == x ** 2 + I
    assert R.convert(ZZ(1), ZZ) == 1 + I
    assert R.convert(R.convert(x), R) == R.convert(x)
    X = R.convert(x)
    Y = QQ.old_poly_ring(x).convert(x)
    assert -1 + I == X ** 2 + I
    assert -1 + I == Y ** 2 + I
    assert R.to_sympy(X) == x
    raises(ValueError, lambda: QQ.old_poly_ring(x) / QQ.old_poly_ring(x, y).ideal(x))
    R = QQ.old_poly_ring(x, order='ilex')
    I = R.ideal(x)
    assert R.convert(1) + I == (R / I).convert(1)