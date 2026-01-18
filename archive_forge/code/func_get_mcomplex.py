from .geodesic_tube_info import GeodesicTubeInfo
from .upper_halfspace_utilities import *
from ..drilling.geometric_structure import add_r13_geometry
from ..drilling.geodesic_tube import add_structures_necessary_for_tube
from ..snap.t3mlite import Mcomplex, simplex
from ..upper_halfspace import pgl2c_to_o13, sl2c_inverse
def get_mcomplex(self):
    if self.mcomplex is None:
        self.mcomplex = Mcomplex(self.manifold)
        add_r13_geometry(self.mcomplex, self.manifold)
        add_structures_necessary_for_tube(self.mcomplex)
        for tet in self.mcomplex.Tetrahedra:
            z = tet.ShapeParameters[simplex.E01]
            vert0 = [tet.ideal_vertices[v] for v in simplex.ZeroSubsimplices[:3]]
            vert1 = symmetric_vertices_for_tetrahedron(z)[:3]
            tet.to_coordinates_in_symmetric_tet = o13_matrix_taking_ideal_vertices_to_ideal_vertices(vert0, vert1)
    return self.mcomplex