import snappy
from sage.all import QQ, PolynomialRing, matrix, prod
import giac_rur
from closed import zhs_exs
import phc_wrapper
def inverse_word(word):
    return word.swapcase()[::-1]