from .polished_reps import ManifoldGroup
from .fundamental_polyhedron import *
def matrix_difference_norm(A, B):
    B = B.change_ring(A.base_ring())
    return max([abs(a - b) for a, b in zip(A.list(), B.list())])