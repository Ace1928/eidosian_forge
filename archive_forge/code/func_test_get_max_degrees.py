from sympy.matrices.dense import Matrix
from sympy.polys.polytools import factor
from sympy.core import symbols
from sympy.tensor.indexed import IndexedBase
from sympy.polys.multivariate_resultants import (DixonResultant,
def test_get_max_degrees():
    """Tests max degrees function."""
    p = x + y
    q = x ** 2 + y ** 3
    h = x ** 2 + y
    dixon = DixonResultant(polynomials=[p, q, h], variables=[x, y])
    dixon_polynomial = dixon.get_dixon_polynomial()
    assert dixon.get_max_degrees(dixon_polynomial) == [1, 2]