import random
from .links_base import CrossingEntryPoint
from ..sage_helper import _within_sage
def alexander(K):
    c = len(K.crossings)
    if c < 100:
        E = Exhaustion(K)
    else:
        E = good_exhaustion(K, max(20, 0.15 * c))
    return E.alexander_polynomial()