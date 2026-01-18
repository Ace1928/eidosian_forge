from .simplex import *
from .tetrahedron import Tetrahedron
from .corner import Corner
from .arrow import Arrow
from .face import Face
from .edge import Edge
from .vertex import Vertex
from .surface import Surface, SpunSurface, ClosedSurface, ClosedSurfaceInCusped
from .perm4 import Perm4, inv
from . import files
from . import linalg
from . import homology
import sys
import random
import io
def isomorphisms_to(self, other, orientation_preserving=False, at_most_one=False):
    """
        Return the list of isomorphisms between the MComplexes M and N.
        If `at_most_one` is `True`, only returns the first one found (but
        still as a list).

        >>> tri_data = [([0,1,0,1], [(2,1,0,3), (0,3,2,1), (2,1,0,3), (0,1,3,2)]),
        ...             ([1,1,0,0], [(1,0,2,3), (1,0,2,3), (0,1,3,2), (0,3,2,1)])]
        >>> M = Mcomplex(tri_data)
        >>> N = Mcomplex(M.isosig())
        >>> isos = M.isomorphisms_to(N); len(isos)
        4
        >>> isos[0]
        {0: [tet0, (0, 2, 1, 3)], 1: [tet1, (0, 2, 1, 3)]}
        >>> len(M.isomorphisms_to(N, orientation_preserving=True))
        2
        >>> M.two_to_three(Arrow(E01, F3, M[0])); M.rebuild()
        True
        >>> len(M), len(N)
        (3, 2)
        >>> M.isomorphisms_to(N)
        []
        >>> F = Mcomplex('m004')
        >>> N.isomorphisms_to(F)
        []
        >>> N = Mcomplex(M.isosig())
        >>> M.isomorphisms_to(N, at_most_one=True)[0]
        {0: [tet1, (0, 2, 3, 1)], 1: [tet2, (0, 2, 3, 1)], 2: [tet0, (0, 3, 1, 2)]}
        >>> M = Mcomplex(tri_data)
        >>> M.two_to_three(Arrow(E01, F3, M[0])); M.two_to_three(Arrow(E01, F3, M[1]))
        True
        True
        >>> M.rebuild()
        >>> len(M) == 4
        True
        >>> N = Mcomplex(M.isosig())
        >>> M.isomorphisms_to(N, at_most_one=True)[0]  # doctest: +NORMALIZE_WHITESPACE
        {0: [tet0, (1, 3, 0, 2)], 1: [tet1, (3, 0, 1, 2)],
         2: [tet3, (2, 0, 3, 1)], 3: [tet2, (3, 1, 2, 0)]}
        """
    M, N = (self, other)
    if not isinstance(N, Mcomplex):
        raise ValueError('The other triangulation must be an Mcomplex')
    if len(M) != len(N):
        return []
    t_M0 = M[0]
    if orientation_preserving:
        if not (M.is_oriented() and N.is_oriented()):
            raise ValueError('Asked for orientation preserving isomorphisms of unoriented triangulations')
        permutations = list(Perm4.A4())
    else:
        permutations = list(Perm4.S4())
    isomorphisms = []
    for t_N0 in N:
        for perm in permutations:
            iso = {k: None for k in range(len(M))}
            iso[0] = [t_N0, perm]
            tet_queue = [t_M0]
            while tet_queue != []:
                t_M = tet_queue.pop()
                t_N = iso[t_M.Index][0]
                perm = iso[t_M.Index][1]
                neighbors_M = [t_M.Neighbor[face] for face in TwoSubsimplices]
                neighbors_N = [t_N.Neighbor[perm.image(face)] for face in TwoSubsimplices]
                gluings_M = [t_M.Gluing[face] for face in TwoSubsimplices]
                gluings_N = [t_N.Gluing[perm.image(face)] for face in TwoSubsimplices]
                maps = [gluings_N[k] * perm * inv(gluings_M[k]) for k in [0, 1, 2, 3]]
                for i in range(len(neighbors_M)):
                    t = neighbors_M[i]
                    s = neighbors_N[i]
                    map = maps[i]
                    if iso[t.Index] is not None:
                        if iso[t.Index][0] != s or iso[t.Index][1].tuple() != map.tuple():
                            iso = {k: None for k in range(len(M))}
                            tet_queue = []
                            break
                    else:
                        iso[t.Index] = [s, map]
                        tet_queue = tet_queue + [t]
            if None not in list(iso.values()):
                isomorphisms.append(iso.copy())
                if at_most_one:
                    return isomorphisms
    return isomorphisms