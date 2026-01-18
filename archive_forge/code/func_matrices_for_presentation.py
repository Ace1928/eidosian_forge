from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def matrices_for_presentation(self, G, match_kernel=False):
    """
        Given the result of M.fundamental_group(...) where M is the
        corresponding SnapPy.Manifold, return the matrices for that
        presentation of the fundamental polyhedron.

        The GeneratorMatrices computed here are for the face-pairing
        presentation with respect to the fundamental polyhedron.
        That presentation can be simplfied by M.fundamental_group(...)
        and this function will compute the matrices for the simplified
        presentation from the GeneratorMatrices.

        If match_kernel is True, it will flip the signs of some of
        the matrices to match the ones in the given G (which were determined
        by the SnapPea kernel).

        This makes the result stable when changing precision (when normalizing
        matrices with determinant -1, sqrt(-1) might jump between i and -i when
        increasing precision).
        """
    num_generators = len(self.mcomplex.GeneratorMatrices) // 2
    matrices = [self.mcomplex.GeneratorMatrices[g + 1] for g in range(num_generators)]
    result = _perform_word_moves(matrices, G)
    if match_kernel:
        return _negate_matrices_to_match_kernel(result, G)
    else:
        return result