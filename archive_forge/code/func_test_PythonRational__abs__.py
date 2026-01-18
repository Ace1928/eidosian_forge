from sympy.polys.domains import PythonRational as QQ
from sympy.testing.pytest import raises
def test_PythonRational__abs__():
    assert abs(QQ(-1, 2)) == QQ(1, 2)
    assert abs(QQ(1, 2)) == QQ(1, 2)