from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def distances_indices_groups(self, points, delta=None, delta_factor=0.05, sign=False):
    """
        Computes the distances from the plane to each of the points. Positive distances are on the side of the
        normal of the plane while negative distances are on the other side. Indices sorting the points from closest
        to furthest is also computed. Grouped indices are also given, for which indices of the distances that are
        separated by less than delta are grouped together. The delta parameter is either set explicitly or taken as
        a fraction (using the delta_factor parameter) of the maximal point distance.

        Args:
            points: Points for which distances are computed
            delta: Distance interval for which two points are considered in the same group.
            delta_factor: If delta is None, the distance interval is taken as delta_factor times the maximal
            point distance.
            sign: Whether to add sign information in the indices sorting the points distances

        Returns:
            Distances from the plane to the points (positive values on the side of the normal to the plane,
            negative values on the other side), as well as indices of the points from closest to furthest and
            grouped indices of distances separated by less than delta. For the sorting list and the grouped
            indices, when the sign parameter is True, items are given as tuples of (index, sign).
        """
    distances, indices = self.distances_indices_sorted(points=points)
    if delta is None:
        delta = delta_factor * np.abs(distances[indices[-1]])
    iends = [ii for ii, idist in enumerate(indices, start=1) if ii == len(distances) or np.abs(distances[indices[ii]]) - np.abs(distances[idist]) > delta]
    if sign:
        indices = [(ii, int(np.sign(distances[ii]))) for ii in indices]
    grouped_indices = [indices[iends[ii - 1]:iend] if ii > 0 else indices[:iend] for ii, iend in enumerate(iends)]
    return (distances, indices, grouped_indices)