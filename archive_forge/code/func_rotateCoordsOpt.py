from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def rotateCoordsOpt(coords, R):
    """
    Rotate the list of points using rotation matrix R

    Args:
        coords: List of points to be rotated
        R: Rotation matrix

    Returns:
        List of rotated points.
    """
    return [np.dot(R, pp) for pp in coords]