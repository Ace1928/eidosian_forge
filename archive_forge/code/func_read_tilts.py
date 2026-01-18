from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def read_tilts(self):
    """
        After compute_tilts() has been called, put the tilt values into an
        array containing the tilt of face 0, 1, 2, 3 of the first tetrahedron,
        ... of the second tetrahedron, ....
        """

    def index_of_face_corner(corner):
        face_index = t3m.simplex.comp(corner.Subsimplex).bit_length() - 1
        return 4 * corner.Tetrahedron.Index + face_index
    tilts = 4 * len(self.mcomplex.Tetrahedra) * [None]
    for face in self.mcomplex.Faces:
        for corner in face.Corners:
            tilts[index_of_face_corner(corner)] = face.Tilt
    return tilts