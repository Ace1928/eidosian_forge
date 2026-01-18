from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def sort_separation(separation):
    """Sort a separation.

    Args:
        separation: Initial separation.

    Returns:
        Sorted list of separation.
    """
    if len(separation[0]) > len(separation[2]):
        return [sorted(separation[2]), sorted(separation[1]), sorted(separation[0])]
    return [sorted(separation[0]), sorted(separation[1]), sorted(separation[2])]