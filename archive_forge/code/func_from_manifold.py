from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.snap.mcomplex_base import *
from snappy.verify.cuspCrossSection import *
from ..upper_halfspace import pgl2c_to_o13, sl2c_inverse
from ..upper_halfspace.ideal_point import ideal_point_to_r13
from .hyperboloid_utilities import *
from .upper_halfspace_utilities import *
from .raytracing_data import *
from math import sqrt
@staticmethod
def from_manifold(manifold, areas=None, insphere_scale=0.05, weights=None):
    if manifold.solution_type() != 'all tetrahedra positively oriented':
        return NonGeometricRaytracingData(t3m.Mcomplex(manifold))
    num_cusps = manifold.num_cusps()
    snappy_trig = Triangulation(manifold)
    snappy_trig.dehn_fill(num_cusps * [(0, 0)])
    c = ComplexCuspCrossSection.fromManifoldAndShapes(manifold, manifold.tetrahedra_shapes('rect'), one_cocycle='develop')
    c.normalize_cusps()
    c.compute_translations()
    c.add_vertex_positions_to_horotriangles()
    c.lift_vertex_positions_of_horotriangles()
    c.move_lifted_vertex_positions_to_zero_first()
    r = IdealRaytracingData(c.mcomplex, manifold)
    z = c.mcomplex.Tetrahedra[0].ShapeParameters[t3m.E01]
    r.RF = z.real().parent()
    r.insphere_scale = r.RF(insphere_scale)
    resolved_areas = num_cusps * [1.0] if areas is None else areas
    r.areas = [r.RF(area) for area in resolved_areas]
    r.peripheral_gluing_equations = snappy_trig.gluing_equations()[snappy_trig.num_tetrahedra():]
    r._add_complex_vertices()
    r._add_R13_vertices()
    r._add_O13_matrices_to_faces()
    r._add_R13_planes_to_faces()
    r._add_R13_horosphere_scales_to_vertices()
    r._add_cusp_to_tet_matrices()
    r._add_margulis_tube_ends()
    r._add_inspheres()
    r._add_log_holonomies()
    r._add_cusp_triangle_vertex_positions()
    r.add_weights(weights)
    return r