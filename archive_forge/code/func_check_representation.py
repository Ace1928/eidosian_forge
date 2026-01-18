from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def check_representation(self):
    relator_matrices = (self.SL2C(R) for R in self.relators())
    return max((projective_distance(A, identity(A)) for A in relator_matrices))