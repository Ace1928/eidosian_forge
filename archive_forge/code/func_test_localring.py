from sympy.polys.domains import QQ, ZZ
from sympy.polys.polyerrors import ExactQuotientFailed, CoercionFailed, NotReversible
from sympy.abc import x, y
from sympy.testing.pytest import raises
def test_localring():
    Qxy = QQ.old_frac_field(x, y)
    R = QQ.old_poly_ring(x, y, order='ilex')
    X = R.convert(x)
    Y = R.convert(y)
    assert x in R
    assert 1 / x not in R
    assert 1 / (1 + x) in R
    assert Y in R
    assert X.ring == R
    assert X * (Y ** 2 + 1) / (1 + X) == R.convert(x * (y ** 2 + 1) / (1 + x))
    assert X * y == X * Y
    raises(ExactQuotientFailed, lambda: X / Y)
    raises(ExactQuotientFailed, lambda: x / Y)
    raises(ExactQuotientFailed, lambda: X / y)
    assert X + y == X + Y == R.convert(x + y) == x + Y
    assert X - y == X - Y == R.convert(x - y) == x - Y
    assert X + 1 == R.convert(x + 1)
    assert X ** 2 / X == X
    assert R.from_GlobalPolynomialRing(ZZ.old_poly_ring(x, y).convert(x), ZZ.old_poly_ring(x, y)) == X
    assert R.from_FractionField(Qxy.convert(x), Qxy) == X
    raises(CoercionFailed, lambda: R.from_FractionField(Qxy.convert(x) / y, Qxy))
    raises(ExactQuotientFailed, lambda: X / Y)
    raises(NotReversible, lambda: X.invert())
    assert R._sdm_to_vector(R._vector_to_sdm([X / (X + 1), Y / (1 + X * Y)], R.order), 2) == [X * (1 + X * Y), Y * (1 + X)]