from sympy.matrices import Matrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.fields import field
from sympy.polys.rings import ring
from sympy.polys.solvers import solve_lin_sys, eqs_to_matrix
def test_eqs_to_matrix():
    domain, x1, x2 = ring('x1,x2', QQ)
    eqs_coeff = [{x1: QQ(1), x2: QQ(1)}, {x1: QQ(2), x2: QQ(-1)}]
    eqs_rhs = [QQ(-5), QQ(0)]
    M = eqs_to_matrix(eqs_coeff, eqs_rhs, [x1, x2], QQ)
    assert M.to_Matrix() == Matrix([[1, 1, 5], [2, -1, 0]])