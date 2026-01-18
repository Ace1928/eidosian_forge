from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def negative_coordinates(self):
    return [l for l in self.vector if l < 0]