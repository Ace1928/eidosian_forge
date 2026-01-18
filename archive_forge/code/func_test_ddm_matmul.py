from sympy.testing.pytest import raises
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.dense import (
from sympy.polys.matrices.exceptions import (
def test_ddm_matmul():
    a = [[1, 2], [3, 4]]
    ddm_imul(a, 2)
    assert a == [[2, 4], [6, 8]]
    a = [[1, 2], [3, 4]]
    ddm_imul(a, 0)
    assert a == [[0, 0], [0, 0]]