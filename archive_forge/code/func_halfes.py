import snappy
from sage.all import QQ, PolynomialRing, matrix, prod
import giac_rur
from closed import zhs_exs
import phc_wrapper
def halfes(word):
    a = len(word) // 2
    return (word[:a], inverse_word(word[a:]))