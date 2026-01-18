import itertools
from functools import reduce
from sympy.core import Dummy, ilcm, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve
def constant_system(A, u, DE):
    """
    Generate a system for the constant solutions.

    Explanation
    ===========

    Given a differential field (K, D) with constant field C = Const(K), a Matrix
    A, and a vector (Matrix) u with coefficients in K, returns the tuple
    (B, v, s), where B is a Matrix with coefficients in C and v is a vector
    (Matrix) such that either v has coefficients in C, in which case s is True
    and the solutions in C of Ax == u are exactly all the solutions of Bx == v,
    or v has a non-constant coefficient, in which case s is False Ax == u has no
    constant solution.

    This algorithm is used both in solving parametric problems and in
    determining if an element a of K is a derivative of an element of K or the
    logarithmic derivative of a K-radical using the structure theorem approach.

    Because Poly does not play well with Matrix yet, this algorithm assumes that
    all matrix entries are Basic expressions.
    """
    if not A:
        return (A, u)
    Au = A.row_join(u)
    Au, _ = Au.rref()
    A, u = (Au[:, :-1], Au[:, -1])
    D = lambda x: derivation(x, DE, basic=True)
    for j, i in itertools.product(range(A.cols), range(A.rows)):
        if A[i, j].expr.has(*DE.T):
            Ri = A[i, :]
            DAij = D(A[i, j])
            Rm1 = Ri.applyfunc(lambda x: D(x) / DAij)
            um1 = D(u[i]) / DAij
            Aj = A[:, j]
            A = A - Aj * Rm1
            u = u - Aj * um1
            A = A.col_join(Rm1)
            u = u.col_join(Matrix([um1], u.gens))
    return (A, u)