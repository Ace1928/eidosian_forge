from ..sage_helper import _within_sage
from ..pari import pari, prec_dec_to_bits, prec_bits_to_dec, Gen
def float_to_pari(x, dec_prec):
    return pari(0) if x == 0 else pari(x).precision(dec_prec)