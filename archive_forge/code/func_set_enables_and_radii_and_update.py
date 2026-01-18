from .geodesic_tube_info import GeodesicTubeInfo
from .upper_halfspace_utilities import *
from ..drilling.geometric_structure import add_r13_geometry
from ..drilling.geodesic_tube import add_structures_necessary_for_tube
from ..snap.t3mlite import Mcomplex, simplex
from ..upper_halfspace import pgl2c_to_o13, sl2c_inverse
def set_enables_and_radii_and_update(self, enables, radii):
    success = True
    if not self.geodesic_tube_infos:
        return success
    self.data_heads = []
    self.data_tails = []
    self.data_indices = []
    self.data_radius_params = []
    self.data_offsets = []
    tets_to_data = [[] for i in range(self.num_tetrahedra)]
    for i, (enable, radius, geodesic_tube) in enumerate(zip(enables, radii, self.geodesic_tube_infos)):
        if enable:
            radius = self.RF(radius)
            tets_and_endpoints, safe_radius = geodesic_tube.compute_tets_and_R13_endpoints_and_radius_for_tube(radius)
            if safe_radius < radius:
                success = False
            radius_param = safe_radius.cosh() ** 2 / 2
            for tet, endpoints in tets_and_endpoints:
                tets_to_data[tet].append((endpoints, i, radius_param))
    for data in tets_to_data:
        self.data_offsets.append(len(self.data_heads))
        for (head, tail), i, radius_param in data:
            self.data_heads.append(head)
            self.data_tails.append(tail)
            self.data_indices.append(i)
            self.data_radius_params.append(radius_param)
    self.data_offsets.append(len(self.data_heads))
    return success