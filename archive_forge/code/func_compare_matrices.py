from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def compare_matrices(Mats0, Mats1):
    return max([projective_distance(A, B) for A, B in zip(Mats0, Mats1)])