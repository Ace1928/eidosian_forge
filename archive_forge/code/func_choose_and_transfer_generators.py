from .mcomplex_base import *
from .t3mlite import simplex
def choose_and_transfer_generators(self, compute_corners, centroid_at_origin):
    self.snappyTriangulation._choose_generators(compute_corners, centroid_at_origin)
    self.add_choose_generators_info(self.snappyTriangulation._choose_generators_info())