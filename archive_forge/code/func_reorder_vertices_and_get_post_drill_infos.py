from .geodesic_info import GeodesicInfo
from .geometric_structure import Filling, FillingMatrix
from ..snap.t3mlite import Mcomplex, simplex
from typing import Tuple, Optional, Sequence
def reorder_vertices_and_get_post_drill_infos(mcomplex: Mcomplex) -> Sequence[CuspPostDrillInfo]:
    cusp_vertices_dict = {}
    finite_vertices = []
    for vert in mcomplex.Vertices:
        c = vert.Corners[0]
        post_drill_info = c.Tetrahedron.post_drill_infos[c.Subsimplex]
        vert.post_drill_info = post_drill_info
        if post_drill_info.index is None:
            finite_vertices.append(vert)
        else:
            cusp_vertices_dict[post_drill_info.index] = vert
    cusp_vertices = [cusp_vertices_dict[i] for i in range(len(cusp_vertices_dict))]
    mcomplex.Vertices = cusp_vertices + finite_vertices
    return [v.post_drill_info for v in cusp_vertices]