import itertools
from functools import reduce
from sympy.core import Dummy, ilcm, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve
def real_imag(ba, bd, gen):
    """
    Helper function, to get the real and imaginary part of a rational function
    evaluated at sqrt(-1) without actually evaluating it at sqrt(-1).

    Explanation
    ===========

    Separates the even and odd power terms by checking the degree of terms wrt
    mod 4. Returns a tuple (ba[0], ba[1], bd) where ba[0] is real part
    of the numerator ba[1] is the imaginary part and bd is the denominator
    of the rational function.
    """
    bd = bd.as_poly(gen).as_dict()
    ba = ba.as_poly(gen).as_dict()
    denom_real = [value if key[0] % 4 == 0 else -value if key[0] % 4 == 2 else 0 for key, value in bd.items()]
    denom_imag = [value if key[0] % 4 == 1 else -value if key[0] % 4 == 3 else 0 for key, value in bd.items()]
    bd_real = sum((r for r in denom_real))
    bd_imag = sum((r for r in denom_imag))
    num_real = [value if key[0] % 4 == 0 else -value if key[0] % 4 == 2 else 0 for key, value in ba.items()]
    num_imag = [value if key[0] % 4 == 1 else -value if key[0] % 4 == 3 else 0 for key, value in ba.items()]
    ba_real = sum((r for r in num_real))
    ba_imag = sum((r for r in num_imag))
    ba = ((ba_real * bd_real + ba_imag * bd_imag).as_poly(gen), (ba_imag * bd_real - ba_real * bd_imag).as_poly(gen))
    bd = (bd_real * bd_real + bd_imag * bd_imag).as_poly(gen)
    return (ba[0], ba[1], bd)