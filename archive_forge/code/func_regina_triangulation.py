from .simplex import *
from .tetrahedron import Tetrahedron
from .corner import Corner
from .arrow import Arrow
from .face import Face
from .edge import Edge
from .vertex import Vertex
from .surface import Surface, SpunSurface, ClosedSurface, ClosedSurfaceInCusped
from .perm4 import Perm4, inv
from . import files
from . import linalg
from . import homology
import sys
import random
import io
def regina_triangulation(self):
    """
        >>> M = Mcomplex('K14n1234')
        >>> try:
        ...     T = M.regina_triangulation()
        ...     assert M.isosig() == T.isoSig()
        ... except ImportError:
        ...     pass
        """
    try:
        import regina
    except ImportError:
        raise ImportError('Regina module not available')
    T = regina.Triangulation3()
    regina_tets = {tet: T.newTetrahedron() for tet in self}
    self.rebuild()
    for face in self.Faces:
        if face.IntOrBdry == 'int':
            corner = face.Corners[0]
            tet0 = corner.Tetrahedron
            face0 = corner.Subsimplex
            tet1 = tet0.Neighbor[face0]
            perm = tet0.Gluing[face0]
            r_tet0 = regina_tets[tet0]
            r_tet1 = regina_tets[tet1]
            r_face = FaceIndex[face0]
            r_perm = regina.Perm4(*perm.tuple())
            r_tet0.join(r_face, r_tet1, r_perm)
    return T