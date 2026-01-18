from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def move_lifted_vertex_positions_to_zero_first(self):
    """
        Shift the lifted vertex positions such that the one associated
        to the first vertex when developing the incomplete cusp is
        zero. This makes the values we obtain more stable when
        changing the Dehn-surgery parameters.
        """
    for cusp in self.mcomplex.Vertices:
        if not cusp.is_complete:
            ComplexCuspCrossSection._move_lifted_vertex_positions_cusp(cusp)