from .gimbalLoopFinder import GimbalLoopFinder
from .truncatedComplex import TruncatedComplex
from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import matrix, prod, RealDoubleField, pi
def assert_vertex_gram_matrices_valid(self):
    for G_adj in self.hyperbolic_structure.vertex_gram_adjoints:
        _assert_vertex_gram_adjoint_valid(G_adj)
    for G in self.hyperbolic_structure.vertex_gram_matrices:
        _assert_vertex_gram_matrix_valid(G)