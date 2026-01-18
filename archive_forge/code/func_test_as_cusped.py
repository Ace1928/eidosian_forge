import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def test_as_cusped(manifold):
    import giac_rur
    for obs in manifold.ptolemy_generalized_obstruction_classes(2):
        I = extended_ptolemy_equations(manifold, obs)
        R = I.ring()
        M, L = (R('M'), R('L'))
        I = I + [M - 1, L - 1]
        if I.dimension() == 0:
            print(giac_rur.rational_univariate_representation(I))