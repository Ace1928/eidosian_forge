from ..sage_helper import _within_sage, sage_method
from .cuspCrossSection import RealCuspCrossSection
from .squareExtensions import find_shapes_as_complex_sqrt_lin_combinations
from . import verifyHyperbolicity
from . import exceptions
from ..exceptions import SnapPeaFatalError
from ..snap import t3mlite as t3m
def index_of_face_corner(corner):
    face_index = t3m.simplex.comp(corner.Subsimplex).bit_length() - 1
    return 4 * corner.Tetrahedron.Index + face_index