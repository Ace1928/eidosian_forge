from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def alex_poly_of_induced_rep(p, knot_exterior, A, chi):
    """
    When A is the matrix generating an irreducible representation V
    that appears with multiplicity 1 in H_1(B_p, F_q) and chi: V -> F_q
    is a homomorphism, computes the (reduced) twisted alexander
    polynomial of Herald-Kirk-Livingston.

    Here is the example from Section 10.3 of [HKL].  When comparing
    with the original, note the polynomnial coeffs in Q(zeta_5) are
    not in the usual Q-basis for this field 1, z, z^2, z^3 where z =
    zeta_5::

       sage: M = Manifold('K12n132')
       sage: A = matrix(GF(5), [[0, 4], [1, 4]])
       sage: chi = lambda v:v[0] + 3*v[1]
       sage: alex = alex_poly_of_induced_rep(3, M, A, chi)
       sage: t = alex.parent().gen()
       sage: quo, rem = alex.quo_rem(t - 1)
       sage: 5*quo/quo.leading_coefficient()
       5*t^3 + (10*z^3 + 14*z^2 + 14*z + 12)*t^2 + (-4*z^2 - 14*z - 2)*t + 5

    Here is their example 10.6::

       sage: M = Manifold('K12n224')
       sage: A3, A5 = matrix(GF(7), [[4]]), matrix(GF(7), [[2]])
       sage: A3.charpoly(), A5.charpoly()
       (x + 3, x + 5)
       sage: -alex_poly_of_induced_rep(3, M, A3, lambda v:3*v[0])
       t^4 + (-4*z^4 - 4*z^2 - 4*z - 5)*t^3 + 6*t^2 + (4*z^4 + 4*z^2 + 4*z - 1)*t + 1
       sage: -alex_poly_of_induced_rep(3, M, A5, lambda v:v[0])
       t^4 + (4*z^4 + 4*z^2 + 4*z - 1)*t^3 + 6*t^2 + (-4*z^4 - 4*z^2 - 4*z - 5)*t + 1
    """
    G = knot_exterior.fundamental_group()
    rho = cyclic_rep(G, A)
    n = rho.dim
    C = rho.twisted_cochain_complex()
    if C.homology(1).dimension() != n:
        raise ValueError('Multiplicity of V is not 1')
    d0, d1 = (C.differential(0), C.differential(1))
    B1 = d0.column_space()
    Z1 = d1.right_kernel()
    cocycle = next((z for z in Z1.basis() if z not in B1))
    alpha = induced_rep_from_twisted_cocycle(p, rho, chi, cocycle)
    ans = twisted_alexander_polynomial(alpha, reduced=True)
    assert poly_involution(ans) == ans
    return ans