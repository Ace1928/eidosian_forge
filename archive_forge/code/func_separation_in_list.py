from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def separation_in_list(separation_indices, separation_indices_list):
    """
    Checks if the separation indices of a plane are already in the list

    Args:
        separation_indices: list of separation indices (three arrays of integers)
        separation_indices_list: list of the list of separation indices to be compared to

    Returns:
        bool: True if the separation indices are already in the list.
    """
    sorted_separation = sort_separation(separation_indices)
    for sep in separation_indices_list:
        if len(sep[1]) == len(sorted_separation[1]) and np.allclose(sorted_separation[1], sep[1]):
            return True
    return False