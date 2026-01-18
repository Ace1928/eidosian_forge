from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def is_nongeneric(self):
    zeros_s = self.start.zero_coordinates()
    zeros_e = self.end.zero_coordinates()
    if len(zeros_s) > 1 or len(zeros_e) > 1:
        return True
    if len(set(zeros_s) & set(zeros_e)) > 0:
        return True
    return False