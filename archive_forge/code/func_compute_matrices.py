from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def compute_matrices(self, normalize_matrices=False):
    """
        Assuming positions were assigned to the vertices, adds
        GeneratorMatrices to the Mcomplex which assigns a matrix to each
        generator.

        Compute generator matrices:

        >>> M = Manifold("s776")
        >>> F = FundamentalPolyhedronEngine.from_manifold_and_shapes(
        ...      M, M.tetrahedra_shapes('rect'), normalize_matrices = True)
        >>> generatorMatrices = F.mcomplex.GeneratorMatrices

        Given a letter such as 'a' or 'A', return matrix for corresponding
        generator:

        >>> def letterToMatrix(l, generatorMatrices):
        ...     g = ord(l.lower()) - ord('a') + 1
        ...     if l.isupper():
        ...         g = -g
        ...     return generatorMatrices[g]

        Check that relations are fulfilled up to sign:

        >>> def p(L): return reduce(lambda x, y: x * y, L)
        >>> def close_to_identity(m, epsilon = 1e-12):
        ...     return abs(m[(0,0)] - 1) < epsilon and abs(m[(1,1)] - 1) < epsilon and abs(m[(0,1)]) < epsilon and abs(m[(1,0)]) < epsilon
        >>> def close_to_pm_identity(m, epsilon = 1e-12):
        ...     return close_to_identity(m, epsilon) or close_to_identity(-m, epsilon)
        >>> G = M.fundamental_group(simplify_presentation = False)
        >>> for rel in G.relators():
        ...     close_to_pm_identity(p([letterToMatrix(l, generatorMatrices) for l in rel]))
        True
        True
        True
        True

        """
    z = self.mcomplex.Tetrahedra[0].ShapeParameters[simplex.E01]
    CF = z.parent()
    self.mcomplex.GeneratorMatrices = {0: matrix([[CF(1), CF(0)], [CF(0), CF(1)]])}
    for g, pairings in self.mcomplex.Generators.items():
        if g > 0:
            m = _compute_pairing_matrix(pairings[0])
            if normalize_matrices:
                m = m / m.det().sqrt()
            self.mcomplex.GeneratorMatrices[g] = m
            self.mcomplex.GeneratorMatrices[-g] = _adjoint2(m)