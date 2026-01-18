from .polished_reps import ManifoldGroup
from .fundamental_polyhedron import *
def contains_zero(A):
    return all((x.contains_zero() for x in A.list()))