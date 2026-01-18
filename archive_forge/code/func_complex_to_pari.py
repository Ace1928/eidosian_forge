from ..sage_helper import _within_sage
from ..pari import pari, prec_dec_to_bits, prec_bits_to_dec, Gen
def complex_to_pari(z, dec_prec):
    return pari.complex(float_to_pari(z.real, dec_prec), float_to_pari(z.imag, dec_prec))