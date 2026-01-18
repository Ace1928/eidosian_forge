from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
@classmethod
def from_npoints_maximum_distance(cls, points) -> Self:
    """Initializes plane from a list of points using a max distance fitting procedure.

        Args:
            points: List of points.

        Returns:
            Plane.
        """
    convex_hull = ConvexHull(points)
    heights = []
    ipoints_heights = []
    for idx, _simplex in enumerate(convex_hull.simplices):
        cc = convex_hull.equations[idx]
        plane = Plane.from_coefficients(cc[0], cc[1], cc[2], cc[3])
        distances = [plane.distance_to_point(pp) for pp in points]
        ipoint_height = np.argmax(distances)
        heights.append(distances[ipoint_height])
        ipoints_heights.append(ipoint_height)
    imin_height = np.argmin(heights)
    normal_vector = convex_hull.equations[imin_height, 0:3]
    cc = convex_hull.equations[imin_height]
    highest_point = points[ipoints_heights[imin_height]]
    middle_point = (Plane.from_coefficients(cc[0], cc[1], cc[2], cc[3]).projectionpoints([highest_point])[0] + highest_point) / 2
    dd = -np.dot(normal_vector, middle_point)
    return cls(np.array([normal_vector[0], normal_vector[1], normal_vector[2], dd], float))