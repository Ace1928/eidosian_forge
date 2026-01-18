from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def on_boundary(self):
    return len(self._zero_coordinates) > 0