from sympy.testing.pytest import raises
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.dense import (
from sympy.polys.matrices.exceptions import (
def test_ddm_transpose():
    a = [[1, 2], [3, 4]]
    assert ddm_transpose(a) == [[1, 3], [2, 4]]