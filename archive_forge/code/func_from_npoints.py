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
def from_npoints(cls, points, best_fit='least_square_distance') -> Self:
    """Initializes plane from a list of points.

        If the number of points is larger than 3, will use a least square fitting or max distance fitting.

        Args:
            points: List of points.
            best_fit: Type of fitting procedure for more than 3 points.

        Returns:
            Plane
        """
    if len(points) == 2:
        return cls.from_2points_and_origin(points[0], points[1])
    if len(points) == 3:
        return cls.from_3points(points[0], points[1], points[2])
    if best_fit == 'least_square_distance':
        return cls.from_npoints_least_square_distance(points)
    if best_fit == 'maximum_distance':
        return cls.from_npoints_maximum_distance(points)
    raise ValueError('Cannot initialize Plane.')