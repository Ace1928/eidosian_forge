from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def pick_exact_and_var_edges(self):
    num_edges = len(self.mcomplex.Edges)
    num_verts = len(self.mcomplex.Vertices)
    J = self.jacobian().change_ring(RealDoubleField())
    self.exact_edges, self.var_edges = _find_rows_and_columns_for_full_rank_submatrix(J, num_edges - 3 * num_verts)