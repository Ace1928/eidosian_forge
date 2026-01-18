from sympy.testing.pytest import raises
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.matrices.dense import Matrix
from sympy.simplify import simplify
from sympy.tensor.array import Array
from sympy.tensor.array.dense_ndim_array import (
from sympy.tensor.array.sparse_ndim_array import (
from sympy.abc import x, y
def test_issue_17851():
    for array_type in array_types:
        A = array_type([])
        assert isinstance(A, array_type)
        assert A.shape == (0,)
        assert list(A) == []