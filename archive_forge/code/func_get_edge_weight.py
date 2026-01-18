from .simplex import *
from .tetrahedron import Tetrahedron
import sys
from .linalg import Vector, Matrix
def get_edge_weight(self, edge):
    j = edge.Index
    return self.EdgeWeights[j]