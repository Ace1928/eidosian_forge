from sympy.testing.pytest import raises
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.matrices.dense import Matrix
from sympy.simplify import simplify
from sympy.tensor.array import Array
from sympy.tensor.array.dense_ndim_array import (
from sympy.tensor.array.sparse_ndim_array import (
from sympy.abc import x, y
def test_issue_18361():
    A = Array([sin(2 * x) - 2 * sin(x) * cos(x)])
    B = Array([sin(x) ** 2 + cos(x) ** 2, 0])
    C = Array([(x + x ** 2) / (x * sin(y) ** 2 + x * cos(y) ** 2), 2 * sin(x) * cos(x)])
    assert simplify(A) == Array([0])
    assert simplify(B) == Array([1, 0])
    assert simplify(C) == Array([x + 1, sin(2 * x)])