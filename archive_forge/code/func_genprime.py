from sympy.polys.galoistools import gf_from_dict, gf_factor_sqf
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
def genprime(n, K):
    return K(nextprime(int((2 ** n * pi).evalf())))