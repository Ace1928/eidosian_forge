from sympy.polys.domains import PythonRational as QQ
from sympy.testing.pytest import raises
def test_PythonRational__float__():
    assert float(QQ(-1, 2)) == -0.5
    assert float(QQ(1, 2)) == 0.5