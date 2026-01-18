from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def make_epsilon(A):
    prec = A.base_ring().precision()
    RR = RealField(prec)
    return RR(2) ** (RR(-0.6) * prec)