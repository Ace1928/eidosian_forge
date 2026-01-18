from sympy.polys.domains import PythonRational as QQ
from sympy.testing.pytest import raises
def test_PythonRational__eq__():
    assert (QQ(1, 2) == QQ(1, 2)) is True
    assert (QQ(1, 2) != QQ(1, 2)) is False
    assert (QQ(1, 2) == QQ(1, 3)) is False
    assert (QQ(1, 2) != QQ(1, 3)) is True