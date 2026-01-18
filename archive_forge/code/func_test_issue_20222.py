from sympy.testing.pytest import raises
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.matrices.dense import Matrix
from sympy.simplify import simplify
from sympy.tensor.array import Array
from sympy.tensor.array.dense_ndim_array import (
from sympy.tensor.array.sparse_ndim_array import (
from sympy.abc import x, y
def test_issue_20222():
    A = Array([[1, 2], [3, 4]])
    B = Matrix([[1, 2], [3, 4]])
    raises(TypeError, lambda: A - B)