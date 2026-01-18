from __future__ import annotations
import numpy as np
from .polytopes import ConvexPolytopeData, PolytopeData, manual_get_vertex, polytope_has_element
def get_augmented_coordinate(target_coordinate, strengths):
    """
    Assembles a coordinate in the system used by `xx_region_polytope`.
    """
    *strengths, beta = strengths
    strengths = sorted(strengths + [0, 0])
    interaction_coordinate = [sum(strengths), strengths[-1], strengths[-2], beta]
    return [*target_coordinate, *interaction_coordinate]