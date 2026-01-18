from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def add_to_list_of_pairs(polymod, exponent):
    if polymod - 1 == 0:
        return
    if polymod + 1 == 0:
        polymod = -1
    for i, (p, e) in enumerate(reduced_polymod_exponent_pairs):
        if p - polymod == 0:
            reduced_polymod_exponent_pairs[i] = (p, e + exponent)
            return
    reduced_polymod_exponent_pairs.append((polymod, exponent))