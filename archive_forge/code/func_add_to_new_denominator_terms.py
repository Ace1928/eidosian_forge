from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def add_to_new_denominator_terms(polymod, exponent):
    for i, (p, e) in enumerate(new_denominator_terms):
        if p - polymod == 0:
            new_denominator_terms[i] = (p, min(e, exponent))
            return
    new_denominator_terms.append((polymod, exponent))