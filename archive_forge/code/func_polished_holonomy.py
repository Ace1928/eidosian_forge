from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def polished_holonomy(manifold, bits_prec=100, fundamental_group_args=[], lift_to_SL2=True, ignore_solution_type=False, dec_prec=None, match_kernel=True):
    """
    Return the fundamental group of M equipt with a high-precision version of the
    holonomy representation::

        sage: M = Manifold('m004')
        sage: G = M.polished_holonomy()
        sage: G('a').trace()
        1.5000000000000000000000000000 - 0.86602540378443864676372317075*I
        sage: G = M.polished_holonomy(bits_prec=1000)
        sage: G('a').trace().parent()
        Complex Field with 1000 bits of precision
    """
    M = manifold
    if dec_prec:
        bits_prec = None
        error = 10 ** (-dec_prec * 0.8)
    else:
        error = 2 ** (-bits_prec * 0.8)
    shapes = M.tetrahedra_shapes('rect', bits_prec=bits_prec, dec_prec=dec_prec)
    G = M.fundamental_group(*fundamental_group_args)
    f = FundamentalPolyhedronEngine.from_manifold_and_shapes(M, shapes, normalize_matrices=True, match_kernel=match_kernel)
    mats = f.matrices_for_presentation(G, match_kernel=True)
    clean_mats = [clean_matrix(A, error=error, prec=bits_prec) for A in mats]
    PG = ManifoldGroup(G.generators(), G.relators(), G.peripheral_curves(), clean_mats)
    if lift_to_SL2:
        PG.lift_to_SL2C()
    else:
        assert PG.is_projective_representation()
    return PG