from ..pari import pari
import string
from itertools import combinations, combinations_with_replacement, product
def rel2(j):
    """Generates type 2 relations for generators (words) i,p0,p1,p2,p3"""
    [i, [p0, p1, p2, p3]] = j
    return mult_traceless(i, p0) * s3(p1, p2, p3) - mult_traceless(i, p1) * s3(p0, p2, p3) + mult_traceless(i, p2) * s3(p0, p1, p3) - mult_traceless(i, p3) * s3(p0, p1, p2)