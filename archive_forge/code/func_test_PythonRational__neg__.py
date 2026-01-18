from sympy.polys.domains import PythonRational as QQ
from sympy.testing.pytest import raises
def test_PythonRational__neg__():
    assert -QQ(-1, 2) == QQ(1, 2)
    assert -QQ(1, 2) == QQ(-1, 2)