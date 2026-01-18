from sympy.matrices.dense import Matrix
from sympy.polys.polytools import factor
from sympy.core import symbols
from sympy.tensor.indexed import IndexedBase
from sympy.polys.multivariate_resultants import (DixonResultant,
def test_macaulay_example_one():
    """Tests the Macaulay for example from [Bruce97]_"""
    x, y, z = symbols('x, y, z')
    a_1_1, a_1_2, a_1_3 = symbols('a_1_1, a_1_2, a_1_3')
    a_2_2, a_2_3, a_3_3 = symbols('a_2_2, a_2_3, a_3_3')
    b_1_1, b_1_2, b_1_3 = symbols('b_1_1, b_1_2, b_1_3')
    b_2_2, b_2_3, b_3_3 = symbols('b_2_2, b_2_3, b_3_3')
    c_1, c_2, c_3 = symbols('c_1, c_2, c_3')
    f_1 = a_1_1 * x ** 2 + a_1_2 * x * y + a_1_3 * x * z + a_2_2 * y ** 2 + a_2_3 * y * z + a_3_3 * z ** 2
    f_2 = b_1_1 * x ** 2 + b_1_2 * x * y + b_1_3 * x * z + b_2_2 * y ** 2 + b_2_3 * y * z + b_3_3 * z ** 2
    f_3 = c_1 * x + c_2 * y + c_3 * z
    mac = MacaulayResultant([f_1, f_2, f_3], [x, y, z])
    assert mac.degrees == [2, 2, 1]
    assert mac.degree_m == 3
    assert mac.monomial_set == [x ** 3, x ** 2 * y, x ** 2 * z, x * y ** 2, x * y * z, x * z ** 2, y ** 3, y ** 2 * z, y * z ** 2, z ** 3]
    assert mac.monomials_size == 10
    assert mac.get_row_coefficients() == [[x, y, z], [x, y, z], [x * y, x * z, y * z, z ** 2]]
    matrix = mac.get_matrix()
    assert matrix.shape == (mac.monomials_size, mac.monomials_size)
    assert mac.get_submatrix(matrix) == Matrix([[a_1_1, a_2_2], [b_1_1, b_2_2]])