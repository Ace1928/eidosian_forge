from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.upper_halfspace import pgl2c_to_o13
from .hyperboloid_utilities import *
from .raytracing_data import *
@staticmethod
def from_triangulation(triangulation, weights=None):
    if not _within_sage:
        raise Exception('Only supported within SageMath :(')
    hyperbolic_structure = compute_approx_hyperbolic_structure_orb(triangulation)
    hyperbolic_structure.pick_exact_and_var_edges()
    hyperbolic_structure = polish_approx_hyperbolic_structure(hyperbolic_structure, bits_prec=212)
    r = FiniteRaytracingData(hyperbolic_structure)
    r.RF = hyperbolic_structure.edge_lengths[0].parent()
    r._compute_matrices(hyperbolic_structure)
    r._compute_tet_vertices()
    r._compute_edge_ends()
    r._compute_planes()
    r._compute_face_pairings()
    r.add_weights(weights)
    return r