from sympy.matrices.dense import Matrix
from sympy.polys.polytools import factor
from sympy.core import symbols
from sympy.tensor.indexed import IndexedBase
from sympy.polys.multivariate_resultants import (DixonResultant,
def test_get_dixon_matrix_example_two():
    """Test Dixon's matrix for example from [Palancz08]_."""
    x, y, z = symbols('x, y, z')
    f = x ** 2 + y ** 2 - 1 + z * 0
    g = x ** 2 + z ** 2 - 1 + y * 0
    h = y ** 2 + z ** 2 - 1
    example_two = DixonResultant([f, g, h], [y, z])
    poly = example_two.get_dixon_polynomial()
    matrix = example_two.get_dixon_matrix(poly)
    expr = 1 - 8 * x ** 2 + 24 * x ** 4 - 32 * x ** 6 + 16 * x ** 8
    assert (matrix.det() - expr).expand() == 0