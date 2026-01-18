from ..pari import pari
import string
from itertools import combinations, combinations_with_replacement, product
def rel1(i):
    """Generates type 1 relations for generators (words) i1,i2,i3 j1,j2,j3"""
    [[i1, i2, i3], [j1, j2, j3]] = i
    return s3(i1, i2, i3) * s3(j1, j2, j3) + 18 * det([[mult_traceless(i1, j1), mult_traceless(i1, j2), mult_traceless(i1, j3)], [mult_traceless(i2, j1), mult_traceless(i2, j2), mult_traceless(i2, j3)], [mult_traceless(i3, j1), mult_traceless(i3, j2), mult_traceless(i3, j3)]])