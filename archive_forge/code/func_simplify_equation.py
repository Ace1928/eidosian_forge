import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def simplify_equation(poly):
    """
    Simplifies the given polynomial in three ways:

    1. Cancels any M*m and L*l pairs.

    2. Sets a0 = 1.

    3. Since all variables represent non-zero quantities, divides by
       the gcd of the monomials terms.

    sage: R = PolynomialRing(QQ, ['M', 'L', 'm', 'l', 'a0', 'x', 'y', 'z'])
    sage: simplify_equation(R('5*M*m^2*L*l^3*x*y + 3*M*m*L*l + 11*M^10*m^3*L^5*l^2*z'))
    11*M^7*L^3*z + 5*m*l^2*x*y + 3
    sage: simplify_equation(R('-a0*x + M^7*m^7*x + L^9*l^3*z + a0^2'))
    L^6*z + 1
    sage: simplify_equation(R('M^2*L*a0*x - M*L*y^2*x + M*z^2*x'))
    -L*y^2 + M*L + z^2
    """
    R = poly.parent()
    ans = R.zero()
    try:
        poly = poly.subs(a0=1)
    except:
        poly = poly.subs(c_1100_0=1)
    for coeff, monomial in list(poly):
        e = monomial.exponents()[0]
        M_exp = e[0] - e[2]
        L_exp = e[1] - e[3]
        if M_exp >= 0:
            M_p, M_n = (M_exp, 0)
        else:
            M_p, M_n = (0, -M_exp)
        if L_exp >= 0:
            L_p, L_n = (L_exp, 0)
        else:
            L_p, L_n = (0, -L_exp)
        ans += coeff * R.monomial(M_p, L_p, M_n, L_n, *e[4:])
    ans = ans // gcd([mono for coeff, mono in list(ans)])
    return ans