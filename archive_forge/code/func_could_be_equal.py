from .polished_reps import ManifoldGroup
from .fundamental_polyhedron import *
def could_be_equal(A, B):
    return contains_zero(A - B)