import json
import os
from .mcomplex_with_memory import McomplexWithMemory
from .mcomplex_with_expansion import McomplexWithExpansion
from .mcomplex_with_link import (link_triangulation,
def geodesic_moves(mcomplex):
    """
    For a triangulation of S^3 with 5 or fewer tetrahedra, give 2 -> 3
    and 3 -> 2 moves that turn it into the base triangulation along
    the geodesic path in the Pachner graph.

    >>> data = [([0,3,1,0], [(3,1,2,0),(0,2,1,3),(1,3,0,2),(3,1,2,0)]),
    ...          ([0,2,2,2], [(2,0,3,1),(3,0,1,2),(0,1,3,2),(3,2,0,1)]),
    ...          ([1,1,4,1], [(1,2,3,0),(2,3,1,0),(3,2,0,1),(0,1,3,2)]),
    ...          ([4,4,0,4], [(3,1,2,0),(0,1,3,2),(0,2,1,3),(0,1,3,2)]),
    ...          ([2,3,3,3], [(2,3,1,0),(0,1,3,2),(0,1,3,2),(3,1,2,0)])]
    >>> M = McomplexWithMemory(data)
    >>> transferred = geodesic_moves(M)
    >>> M.perform_moves(transferred)
    >>> M.isosig()
    'cMcabbgdv'
    """
    A = McomplexWithMemory(mcomplex._triangulation_data())
    isosig = A.isosig()
    B = McomplexWithMemory(isosig)
    new_moves = []
    for move in geodesic_map[isosig]:
        iso = B.isomorphisms_to(A, at_most_one=True)[0]
        move_type, B_edge, B_face, B_tet_idx = move
        A_tet, perm = iso[B_tet_idx]
        assert A_tet in A.Tetrahedra
        new_move = (move_type, perm.image(B_edge), perm.image(B_face), A_tet.Index)
        new_moves.append(new_move)
        A.perform_moves([new_move])
        B.perform_moves([move])
    return new_moves