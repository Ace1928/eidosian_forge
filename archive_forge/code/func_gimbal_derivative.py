from .gimbalLoopFinder import GimbalLoopFinder
from .truncatedComplex import TruncatedComplex
from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import matrix, prod, RealDoubleField, pi
def gimbal_derivative(self):
    self.edge_index_to_column_index = {e: i for i, e in enumerate(self.approx_edges)}
    RIF = self.hyperbolic_structure.vertex_gram_matrices[0].base_ring()
    num_rows = 3 * len(self.mcomplex.Vertices)
    num_cols = len(self.approx_edges)
    result = matrix(RIF, num_rows, num_cols)
    for i, gimbal_loop in enumerate(self.gimbal_loops):
        path_matrices = [self.hyperbolic_structure.so3_matrix_for_path(edgePath) for edgeLoop, edgePath in gimbal_loop]
        for j, (edgeLoop, edgePath) in enumerate(gimbal_loop):
            col = self.edge_index_to_column_index[edgeLoop.edge_index]
            m = self._gimbal_derivative_matrix(gimbal_loop, path_matrices, j)
            result[3 * i, col] += m[0, 1]
            result[3 * i + 1, col] += m[0, 2]
            result[3 * i + 2, col] += m[1, 2]
    return result