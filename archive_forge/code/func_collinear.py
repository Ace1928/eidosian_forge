from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def collinear(p1, p2, p3=None, tolerance=0.25):
    """
    Checks if the three points p1, p2 and p3 are collinear or not within a given tolerance. The collinearity is
    checked by computing the area of the triangle defined by the three points p1, p2 and p3. If the area of this
    triangle is less than (tolerance x largest_triangle), then the three points are considered collinear. The
    largest_triangle is defined as the right triangle whose legs are the two smallest distances between the three
     points ie, its area is : 0.5 x (min(|p2-p1|,|p3-p1|,|p3-p2|) x second_min(|p2-p1|,|p3-p1|,|p3-p2|))

    Args:
        p1: First point
        p2: Second point
        p3: Third point (origin [0.0, 0.0, 0.0 if not given])
        tolerance: Area tolerance for the collinearity test (0.25 gives about 0.125 deviation from the line)

    Returns:
        bool: True if the three points are considered as collinear within the given tolerance.
    """
    if p3 is None:
        triangle_area = 0.5 * np.linalg.norm(np.cross(p1, p2))
        dist = np.sort([np.linalg.norm(p2 - p1), np.linalg.norm(p1), np.linalg.norm(p2)])
    else:
        triangle_area = 0.5 * np.linalg.norm(np.cross(p1 - p3, p2 - p3))
        dist = np.sort([np.linalg.norm(p2 - p1), np.linalg.norm(p3 - p1), np.linalg.norm(p3 - p2)])
    largest_triangle_area = 0.5 * dist[0] * dist[1]
    return triangle_area < tolerance * largest_triangle_area