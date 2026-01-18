from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def long_edge(self, tet, v0, v1, v2):
    """
        The matrix that labels a long edge starting at vertex (v0, v1, v2)
        of a doubly truncated simplex corresponding to the ideal tetrahedron
        with index tet.

        This matrix was labeled alpha^{v0v1v2} in Figure 18 of
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711

        It is computed using equation 10.22.

        The resulting matrix is given as a python list of lists.
        """
    key = 'long_edge'
    if key not in self._edge_cache:
        N = self.N()
        m = [[_kronecker_delta(i + j, N - 1) for i in range(N)] for j in range(N)]
        self._edge_cache[key] = m
    return self._edge_cache[key]