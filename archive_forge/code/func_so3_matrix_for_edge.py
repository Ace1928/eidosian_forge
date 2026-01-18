from .verificationError import *
from snappy.snap import t3mlite as t3m
from sage.all import vector, matrix, prod, exp, RealDoubleField, sqrt
import sage.all
def so3_matrix_for_edge(self, e):
    """
        e is a TruncatedComplex.Edge
        """
    if e.subcomplex_type == 'beta':
        eta = self._compute_eta(e.tet_and_perm)
        c = eta.cos()
        s = eta.sin()
        return matrix([[-c, 0, s], [0, -1, 0], [s, 0, c]])
    if e.subcomplex_type == 'gamma':
        theta = self._compute_theta(e.tet_and_perm)
        return HyperbolicStructure.so3_matrix_for_z_rotation(theta)
    raise Exception('Unsupported edge type')