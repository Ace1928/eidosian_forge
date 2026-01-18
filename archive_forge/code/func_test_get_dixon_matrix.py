from sympy.matrices.dense import Matrix
from sympy.polys.polytools import factor
from sympy.core import symbols
from sympy.tensor.indexed import IndexedBase
from sympy.polys.multivariate_resultants import (DixonResultant,
def test_get_dixon_matrix():
    """Test Dixon's resultant for a numerical example."""
    x, y = symbols('x, y')
    p = x + y
    q = x ** 2 + y ** 3
    h = x ** 2 + y
    dixon = DixonResultant([p, q, h], [x, y])
    polynomial = dixon.get_dixon_polynomial()
    assert dixon.get_dixon_matrix(polynomial).det() == 0