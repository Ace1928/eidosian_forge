from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def is_projective_representation(self):
    """
        True if this is an PSL(2,C)-representation, i.e., if multiplying the generators
        in a word yields the identity matrix or its negative.
        """
    rel_images = (self(R) for R in self.relators())
    return all((is_essentially_Id2(M) or is_essentially_Id2(-M) for M in rel_images))