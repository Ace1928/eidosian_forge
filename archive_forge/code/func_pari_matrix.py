from ..sage_helper import _within_sage
from ..pari import pari, prec_dec_to_bits, prec_bits_to_dec, Gen
def pari_matrix(A):
    return pari.matrix(len(A), len(A[0]), [pari(x) for x in sum(A, [])])