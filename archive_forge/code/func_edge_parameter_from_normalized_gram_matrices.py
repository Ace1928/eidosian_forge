from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import vector, RealDoubleField, sqrt
def edge_parameter_from_normalized_gram_matrices(edge, gram_matrices):
    corner = edge.Corners[0]
    s = corner.Subsimplex
    i, j = [k for k in range(4) if s & 1 << k]
    return gram_matrices[corner.Tetrahedron.Index][i][j]