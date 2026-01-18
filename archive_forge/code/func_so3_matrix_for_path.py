from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def so3_matrix_for_path(self, p):
    if p:
        return prod([self.so3_matrix_for_edge(e) for e in p[::-1]])
    else:
        RF = self.vertex_gram_matrices[0].base_ring()
        return matrix.identity(RF, 3)