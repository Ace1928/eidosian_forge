from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def distances_indices_sorted(self, points, sign=False):
    """
        Computes the distances from the plane to each of the points. Positive distances are on the side of the
        normal of the plane while negative distances are on the other side. Indices sorting the points from closest
        to furthest is also computed.

        Args:
            points: Points for which distances are computed
            sign: Whether to add sign information in the indices sorting the points distances

        Returns:
            Distances from the plane to the points (positive values on the side of the normal to the plane,
            negative values on the other side), as well as indices of the points from closest to furthest. For the
            latter, when the sign parameter is True, items of the sorting list are given as tuples of (index, sign).
        """
    distances = [np.dot(self.normal_vector, pp) + self.d for pp in points]
    indices = sorted(range(len(distances)), key=lambda k: np.abs(distances[k]))
    if sign:
        indices = [(ii, int(np.sign(distances[ii]))) for ii in indices]
    return (distances, indices)