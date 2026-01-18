from .hyperboloid_utilities import *
from .ideal_raytracing_data import *
from .finite_raytracing_data import *
from .hyperboloid_navigation import *
from .geodesics import Geodesics
from . import shaders
from snappy.CyOpenGL import SimpleImageShaderWidget
from snappy.SnapPy import vector, matrix
import math
def compute_translation_and_inverse_from_pick_point(self, size, frag_coord, depth):
    RF = self.raytracing_data.RF
    depth = min(depth, _max_depth_for_orbiting)
    view_scale = self.ui_uniform_dict['viewScale'][1]
    perspective_type = self.ui_uniform_dict['perspectiveType'][1]
    x = (frag_coord[0] - 0.5 * size[0]) / min(size[0], size[1])
    y = (frag_coord[1] - 0.5 * size[1]) / min(size[0], size[1])
    if perspective_type == 0:
        scaled_x = 2.0 * view_scale * x
        scaled_y = 2.0 * view_scale * y
        dist = RF(depth).arctanh()
        dir = vector([RF(scaled_x), RF(scaled_y), RF(-1)])
    else:
        if perspective_type == 1:
            scaled_x = view_scale * x
            scaled_y = view_scale * y
            r2 = 0.5 * (scaled_x * scaled_x + scaled_y * scaled_y)
            ray_end = vector([RF(r2 + 1.0 + depth * r2), RF(scaled_x + depth * scaled_x), RF(scaled_y + depth * scaled_y), RF(r2 + depth * (r2 - 1.0))])
        else:
            pt = R13_normalise(vector([RF(1.0), RF(2.0 * x), RF(2.0 * y), RF(0.0)]))
            ray_end = vector([pt[0], pt[1], pt[2], RF(-depth)])
        ray_end = R13_normalise(ray_end)
        dist = ray_end[0].arccosh()
        dir = vector([ray_end[1], ray_end[2], ray_end[3]])
    dir = dir.normalized()
    poincare_dist = (dist / 2).tanh()
    hyp_circumference_up_to_constant = poincare_dist / (1.0 - poincare_dist * poincare_dist)
    speed = min(_max_orbit_speed, _max_linear_camera_speed / max(1e-10, hyp_circumference_up_to_constant))
    return (unit_3_vector_and_distance_to_O13_hyperbolic_translation(dir, dist), unit_3_vector_and_distance_to_O13_hyperbolic_translation(dir, -dist), speed)