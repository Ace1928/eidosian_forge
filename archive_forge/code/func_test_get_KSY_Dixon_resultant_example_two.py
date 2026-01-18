from sympy.matrices.dense import Matrix
from sympy.polys.polytools import factor
from sympy.core import symbols
from sympy.tensor.indexed import IndexedBase
from sympy.polys.multivariate_resultants import (DixonResultant,
def test_get_KSY_Dixon_resultant_example_two():
    """Tests the KSY Dixon resultant for example two"""
    x, y, A = symbols('x, y, A')
    p = x * y + x * A + x - A ** 2 - A + y ** 2 + y
    q = x ** 2 + x * A - x + x * y + y * A - y
    h = x ** 2 + x * y + 2 * x - x * A - y * A - 2 * A
    dixon = DixonResultant([p, q, h], [x, y])
    dixon_poly = dixon.get_dixon_polynomial()
    dixon_matrix = dixon.get_dixon_matrix(dixon_poly)
    D = factor(dixon.get_KSY_Dixon_resultant(dixon_matrix))
    assert D == -8 * A * (A - 1) * (A + 2) * (2 * A - 1) ** 2