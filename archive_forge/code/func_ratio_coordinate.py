from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def ratio_coordinate(self, tet, v0, v1, pt):
    """
        Returns the ratio coordinate for tetrahedron with index tet
        for the edge from v0 to v1 (integers between 0 and 3) and integral
        point pt (quadruple adding up N-1) on the edge.

        See Definition 10.2:
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711

        Note that this definition turned out to have the wrong sign. Multiply
        the result by -1 if v1 < v0 and N is even.
        """
    pt_v0 = [a + _kronecker_delta(v0, i) for i, a in enumerate(pt)]
    pt_v1 = [a + _kronecker_delta(v1, i) for i, a in enumerate(pt)]
    c_pt_v0 = self._coordinate_at_tet_and_point(tet, pt_v0)
    c_pt_v1 = self._coordinate_at_tet_and_point(tet, pt_v1)
    s = (-1) ** pt[v1]
    if v1 < v0 and self.N() % 2 == 0:
        s *= -1
    return s * c_pt_v1 / c_pt_v0