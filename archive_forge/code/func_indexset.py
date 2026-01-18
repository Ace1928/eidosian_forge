from ..sage_helper import _within_sage
from snappy.number import SnapPyNumbers, Number
from itertools import chain
from ..pari import pari, PariError
from .fundamental_polyhedron import Infinity
def indexset(n):
    """The orders of the non-zero bits in the binary expansion of n."""
    i = 0
    result = []
    while True:
        mask = 1 << i
        if n & mask:
            result.append(i)
        if n < mask:
            break
        i += 1
    return result