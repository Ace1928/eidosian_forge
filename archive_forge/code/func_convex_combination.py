from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def convex_combination(self, other, t):
    c0, c1, c2, c3 = (1 - t) * self.vector + t * other.vector
    return BarycentricPoint(c0, c1, c2, c3)