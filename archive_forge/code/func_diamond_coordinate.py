from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def diamond_coordinate(self, tet, v0, v1, v2, pt):
    """
        Returns the diamond coordinate for tetrahedron with index tet
        for the face with vertices v0, v1, v2 (integers between 0 and 3) and
        integral point pt (quadruple adding up to N-2).

        See Definition 10.1:
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711
        """
    pt_v0_v0 = [a + 2 * _kronecker_delta(v0, i) for i, a in enumerate(pt)]
    pt_v0_v1 = [a + _kronecker_delta(v0, i) + _kronecker_delta(v1, i) for i, a in enumerate(pt)]
    pt_v0_v2 = [a + _kronecker_delta(v0, i) + _kronecker_delta(v2, i) for i, a in enumerate(pt)]
    pt_v1_v2 = [a + _kronecker_delta(v1, i) + _kronecker_delta(v2, i) for i, a in enumerate(pt)]
    c_pt_v0_v0 = self._coordinate_at_tet_and_point(tet, pt_v0_v0)
    c_pt_v0_v1 = self._coordinate_at_tet_and_point(tet, pt_v0_v1)
    c_pt_v0_v2 = self._coordinate_at_tet_and_point(tet, pt_v0_v2)
    c_pt_v1_v2 = self._coordinate_at_tet_and_point(tet, pt_v1_v2)
    face = list(set(range(4)) - set([v0, v1, v2]))[0]
    obstruction = self._get_obstruction_variable(face, tet)
    s = PtolemyCoordinates._three_perm_sign(v0, v1, v2)
    return -(obstruction * s * (c_pt_v0_v0 * c_pt_v1_v2) / (c_pt_v0_v1 * c_pt_v0_v2))