from sympy.polys.domains import PythonRational as QQ
from sympy.testing.pytest import raises
def test_PythonRational__hash__():
    assert hash(QQ(0)) == hash(0)
    assert hash(QQ(1)) == hash(1)
    assert hash(QQ(117)) == hash(117)