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
def snappy_triangulation(self, remove_finite_vertices=True):
    """
        >>> Mcomplex('4_1').snappy_manifold().homology()
        Z

        WARNING: Code implicitly assumes all vertex links are orientable.
        """
    tet_to_index = {T: i for i, T in enumerate(self.Tetrahedra)}
    to_cusp_index = {vertex: -1 for vertex in self.Vertices}
    torus_cusps = 0
    for vertex in self.Vertices:
        g = vertex.link_genus()
        if g > 1:
            raise ValueError('Link of vertex has genus more than 1.')
        if g == 1:
            to_cusp_index[vertex] = torus_cusps
            torus_cusps += 1
    tet_data, cusp_indices, peripheral_curves = ([], [], [])
    for tet in self.Tetrahedra:
        neighbors, perms = ([], [])
        for face in TwoSubsimplices:
            if tet.Neighbor[face] is None:
                raise ValueError('SnapPy triangulations cannot have boundary')
            neighbor = tet_to_index[tet.Neighbor[face]]
            perm = tet.Gluing[face].tuple()
            neighbors.append(neighbor)
            perms.append(perm)
        tet_data.append((neighbors, perms))
        cusp_indices.append([to_cusp_index[tet.Class[vert]] for vert in ZeroSubsimplices])
        if hasattr(tet, 'PeripheralCurves'):
            for curve in tet.PeripheralCurves:
                for sheet in curve:
                    one_curve_data = []
                    for v in ZeroSubsimplices:
                        for f in TwoSubsimplices:
                            one_curve_data.append(sheet[v][f])
                    peripheral_curves.append(one_curve_data)
        else:
            for i in range(4):
                peripheral_curves.append(16 * [0])
    M = snappy.Triangulation('empty')
    M._from_tetrahedra_gluing_data(tetrahedra_data=tet_data, num_or_cusps=torus_cusps, num_nonor_cusps=0, cusp_indices=cusp_indices, peripheral_curves=peripheral_curves, remove_finite_vertices=remove_finite_vertices)
    return M