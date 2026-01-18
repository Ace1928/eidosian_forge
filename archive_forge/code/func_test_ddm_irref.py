from sympy.testing.pytest import raises
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.dense import (
from sympy.polys.matrices.exceptions import (
def test_ddm_irref():
    A = []
    Ar = []
    pivots = []
    assert ddm_irref(A) == pivots
    assert A == Ar
    A = [[QQ(0), QQ(1)], [QQ(1), QQ(1)]]
    Ar = [[QQ(1), QQ(0)], [QQ(0), QQ(1)]]
    pivots = [0, 1]
    assert ddm_irref(A) == pivots
    assert A == Ar
    A = [[QQ(1), QQ(2), QQ(1)], [QQ(3), QQ(4), QQ(1)]]
    Ar = [[QQ(1), QQ(0), QQ(-1)], [QQ(0), QQ(1), QQ(1)]]
    pivots = [0, 1]
    assert ddm_irref(A) == pivots
    assert A == Ar
    A = [[QQ(3), QQ(4), QQ(1)], [QQ(1), QQ(2), QQ(1)]]
    Ar = [[QQ(1), QQ(0), QQ(-1)], [QQ(0), QQ(1), QQ(1)]]
    pivots = [0, 1]
    assert ddm_irref(A) == pivots
    assert A == Ar
    A = [[QQ(1), QQ(0)], [QQ(1), QQ(3)], [QQ(0), QQ(1)]]
    Ar = [[QQ(1), QQ(0)], [QQ(0), QQ(1)], [QQ(0), QQ(0)]]
    pivots = [0, 1]
    assert ddm_irref(A) == pivots
    assert A == Ar
    A = [[QQ(1), QQ(0), QQ(1)], [QQ(3), QQ(0), QQ(1)]]
    Ar = [[QQ(1), QQ(0), QQ(0)], [QQ(0), QQ(0), QQ(1)]]
    pivots = [0, 2]
    assert ddm_irref(A) == pivots
    assert A == Ar
    A = [[QQ(0), QQ(1)], [QQ(0), QQ(2)], [QQ(1), QQ(0)]]
    Ar = [[QQ(1), QQ(0)], [QQ(0), QQ(1)], [QQ(0), QQ(0)]]
    pivots = [0, 1]
    assert ddm_irref(A) == pivots
    assert A == Ar