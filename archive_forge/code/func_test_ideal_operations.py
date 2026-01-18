from sympy.polys import QQ, ilex
from sympy.abc import x, y, z
from sympy.testing.pytest import raises
def test_ideal_operations():
    R = QQ.old_poly_ring(x, y)
    I = R.ideal(x)
    J = R.ideal(y)
    S = R.ideal(x * y)
    T = R.ideal(x, y)
    assert not I == J
    assert I == I
    assert I.union(J) == T
    assert I + J == T
    assert I + T == T
    assert not I.subset(T)
    assert T.subset(I)
    assert I.product(J) == S
    assert I * J == S
    assert x * J == S
    assert I * y == S
    assert R.convert(x) * J == S
    assert I * R.convert(y) == S
    assert not I.is_zero()
    assert not J.is_whole_ring()
    assert R.ideal(x ** 2 + 1, x).is_whole_ring()
    assert R.ideal() == R.ideal(0)
    assert R.ideal().is_zero()
    assert T.contains(x * y)
    assert T.subset([x, y])
    assert T.in_terms_of_generators(x) == [R(1), R(0)]
    assert T ** 0 == R.ideal(1)
    assert T ** 1 == T
    assert T ** 2 == R.ideal(x ** 2, y ** 2, x * y)
    assert I ** 5 == R.ideal(x ** 5)