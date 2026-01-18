from sympy.polys.domains import PythonRational as QQ
from sympy.testing.pytest import raises
def test_PythonRational__sub__():
    assert QQ(-1, 2) - QQ(1, 2) == QQ(-1)
    assert QQ(1, 2) - QQ(-1, 2) == QQ(1)
    assert QQ(1, 2) - QQ(1, 2) == QQ(0)
    assert QQ(1, 2) - QQ(3, 2) == QQ(-1)
    assert QQ(3, 2) - QQ(1, 2) == QQ(1)
    assert QQ(3, 2) - QQ(3, 2) == QQ(0)
    assert 1 - QQ(1, 2) == QQ(1, 2)
    assert QQ(1, 2) - 1 == QQ(-1, 2)