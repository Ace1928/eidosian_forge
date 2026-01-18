from sympy.testing.pytest import warns_deprecated_sympy
from sympy.core.symbol import Symbol
from sympy.polys.polytools import Poly
from sympy.matrices import Matrix
from sympy.matrices.normalforms import (
from sympy.polys.domains import ZZ, QQ
from sympy.core.numbers import Integer
def test_issue_23410():
    A = Matrix([[1, 12], [0, 8], [0, 5]])
    H = Matrix([[1, 0], [0, 8], [0, 5]])
    assert hermite_normal_form(A) == H