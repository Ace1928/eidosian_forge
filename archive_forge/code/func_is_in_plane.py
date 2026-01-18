from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def is_in_plane(self, pp, dist_tolerance) -> bool:
    """
        Determines if point pp is in the plane within the tolerance dist_tolerance

        Args:
            pp: point to be tested
            dist_tolerance: tolerance on the distance to the plane within which point pp is considered in the plane

        Returns:
            bool: True if pp is in the plane.
        """
    return np.abs(np.dot(self.normal_vector, pp) + self._coefficients[3]) <= dist_tolerance