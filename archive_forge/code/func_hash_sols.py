import sys, snappy, giac_rur, extended, phc_wrapper, time, gluing
from sage.all import QQ, PolynomialRing, CC, QQbar, macaulay2
def hash_sols(sols):
    return {hash_sol(sol) for sol in sols}