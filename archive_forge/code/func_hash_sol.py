import sys, snappy, giac_rur, extended, phc_wrapper, time, gluing
from sage.all import QQ, PolynomialRing, CC, QQbar, macaulay2
def hash_sol(sol):
    zs = sorted([k for k in sol.keys() if k[0] == 'z'], key=lambda x: int(x[1:]))
    return ';'.join(['%.6f,%.6f' % (sol[z].real, sol[z].imag) for z in zs])