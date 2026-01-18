import sys, snappy, giac_rur, extended, phc_wrapper, time, gluing
from sage.all import QQ, PolynomialRing, CC, QQbar, macaulay2
def ptolemy_rur(manifold):
    return extended.rur_for_dehn_filling(manifold)