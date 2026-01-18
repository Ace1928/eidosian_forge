import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def apoly(manifold, rational_coeff=False, method='sage'):
    """
    Computes the SL(2, C) version of the A-polynomial starting from
    the extended Ptolemy variety.

    By default, uses Sage (which is to say Singular) to eliminate
    variables.  Surprisingly, Macaulay2 is *much* slower.

    sage: M = Manifold('m003')
    sage: I = apoly(M)
    sage: I.gens()
    [M^4*L^2 + M^3*L^2 - M*L^4 - 2*M^2*L^2 - M^3 + M*L^2 + L^2]
    """
    I = extended_ptolemy_equations(manifold)
    R = I.ring()
    if rational_coeff is False:
        F = GF(31991)
        R = R.change_ring(F)
        I = I.change_ring(R)
    to_elim = [R(x) for x in R.variable_names() if x not in ['M', 'L']]
    if method == 'sage':
        return I.elimination_ideal(to_elim)
    elif method == 'M2':
        from sage.all import macaulay2
        I_m2 = macaulay2(I)
        return I_m2.eliminate('{' + repr(to_elim)[1:-1] + '}').to_sage()
    else:
        raise ValueError("method flag should be in ['sage', 'M2']")