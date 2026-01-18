import sys, snappy, giac_rur, extended, phc_wrapper, time, gluing
from sage.all import QQ, PolynomialRing, CC, QQbar, macaulay2
def ptolemy_phc_direct(manifold):
    I = extended.ptolemy_ideal_for_filled(manifold)
    return phc_wrapper.phcpy_direct(I)