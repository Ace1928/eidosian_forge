from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def check_logarithmic_edge_equations_and_positivity(self, NumericalField):
    """
        Check that the shapes have positive imaginary part and that the
        logarithmic gluing equations have small error.

        The shapes are coerced into the field given as argument before the
        logarithm is computed. It can be, e.g., a ComplexIntervalField.
        """
    for edge in self.mcomplex.Edges:
        log_sum = 0
        for tet, perm in edge.embeddings():
            shape = CuspCrossSectionBase._shape_for_edge_embedding(tet, perm)
            numerical_shape = NumericalField(shape)
            log_shape = log(numerical_shape)
            if not log_shape.imag() > 0:
                raise ShapePositiveImaginaryPartNumericalVerifyError(numerical_shape)
            log_sum += log_shape
        twoPiI = NumericalField.pi() * NumericalField(2j)
        if not abs(log_sum - twoPiI) < NumericalField(1e-07):
            raise EdgeEquationLogLiftNumericalVerifyError(log_sum)