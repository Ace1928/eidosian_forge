from ...sage_helper import _within_sage
from .extended_matrix import ExtendedMatrix
def extended_matrix2(self, isOrientationReversing):
    return ExtendedMatrix(self.matrix2(), isOrientationReversing)