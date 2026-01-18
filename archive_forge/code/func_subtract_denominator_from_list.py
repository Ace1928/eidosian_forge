from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def subtract_denominator_from_list(polymod, exponent, l):
    for i, (p, e) in enumerate(l):
        if p - polymod == 0:
            l[i] = (p, e - exponent)
            return