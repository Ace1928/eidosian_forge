import sys, snappy, giac_rur, extended, phc_wrapper, time, gluing
from sage.all import QQ, PolynomialRing, CC, QQbar, macaulay2
def ptolemy_sage(manifold):
    I = extended.ptolemy_ideal_for_filled(manifold)
    return I.variety(QQbar)