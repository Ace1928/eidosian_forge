from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.tetrahedron import Tetrahedron
from ..snap.t3mlite.mcomplex import VERBOSE
from .exceptions import GeneralPositionError
from .rational_linear_algebra import Vector3, QQ
from . import pl_utils
from . import stored_moves
from .mcomplex_with_expansion import McomplexWithExpansion
from .mcomplex_with_memory import McomplexWithMemory
from .barycentric_geometry import (BarycentricPoint, BarycentricArc,
import random
import collections
import time
def link_triangulation(manifold, add_arcs=True, simplify=True, easy_simplify=False, jiggle_limit=100, randomize=0):
    """
    Given the SnapPy manifold of a link exterior, return an Mcomplex
    with the barycentric arcs representing the link.

    >>> KT = link_triangulation(Manifold('K4a1'), simplify=True)
    >>> len(KT)
    13
    >>> KT = link_triangulation(Manifold('L5a1'), simplify=True)
    >>> len(KT) <= 25
    True

    """
    if hasattr(manifold, 'without_hyperbolic_structure'):
        M = manifold.without_hyperbolic_structure()
    else:
        M = manifold.copy()
    n = M.num_cusps()
    if M.cusp_info('is_complete') == n * [True]:
        M.dehn_fill(n * [(1, 0)])
    assert M.cusp_info('is_complete') == n * [False]
    T = M._unsimplified_filled_triangulation(method='layered_and_marked')
    T.simplify(passes_at_fours=jiggle_limit)
    for i in range(randomize):
        T.randomize(passes_at_fours=jiggle_limit)
    MC = McomplexWithMemory(T)
    solid_tori_indices = []
    for i, m in enumerate(T._marked_tetrahedra()):
        if m > 0:
            solid_tori_indices.append((m, i))
    solid_tori_indices.sort()
    solid_tori_tets = [MC[i] for m, i in solid_tori_indices]
    assert len(solid_tori_tets) == n
    MC.invariant_tetrahedra = solid_tori_tets
    if not MC.smash_all_edges():
        return None
    if easy_simplify:
        MC.easy_simplify()
    elif simplify:
        MC.easy_simplify()
        MC.simplify(jiggle_limit=jiggle_limit)
    for tet in solid_tori_tets:
        MC.Tetrahedra.remove(tet)
    MC.Tetrahedra += solid_tori_tets
    MA = McomplexWithLink(MC._triangulation_data())
    MA.name = M.name()
    assert all((is_standard_solid_torus(tet) for tet in MA.Tetrahedra[-n:]))
    if add_arcs:
        add_arcs_to_standard_solid_tori(MA, n)
    return MA