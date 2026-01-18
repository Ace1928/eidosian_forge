from ..sage_helper import _within_sage
from ..pari import pari, prec_dec_to_bits, prec_bits_to_dec, Gen
def gluing_equation_error(eqns, shapes):
    return infinity_norm(gluing_equation_errors(eqns, shapes))