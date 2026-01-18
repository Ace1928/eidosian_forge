from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def term_to_expand_fraction_by(old_terms):
    result = [pair for pair in new_denominator_terms]
    for p, e in old_terms:
        if e < 0:
            subtract_denominator_from_list(p, e, result)
    for p, e in result:
        assert e <= 0
    return prod([p ** (-e) for p, e in result if e < 0])