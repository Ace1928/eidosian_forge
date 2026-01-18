from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def quarter_ellipsis_functions(xx: ArrayLike, yy: ArrayLike) -> dict[str, Callable]:
    """
    Method that creates two quarter-ellipse functions based on points xx and yy. The ellipsis is supposed to
    be aligned with the axes. The two ellipsis pass through the two points xx and yy.

    Args:
        xx: First point
        yy: Second point

    Returns:
        A dictionary with the lower and upper quarter ellipsis functions.
    """
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    if np.any(xx == yy):
        raise RuntimeError('Invalid points for quarter_ellipsis_functions')
    if np.all(xx < yy) or np.all(xx > yy):
        if xx[0] < yy[0]:
            p1 = xx
            p2 = yy
        else:
            p1 = yy
            p2 = xx
        c_lower = np.array([p1[0], p2[1]])
        c_upper = np.array([p2[0], p1[1]])
        b2 = (p2[1] - p1[1]) ** 2
    else:
        if xx[0] < yy[0]:
            p1 = xx
            p2 = yy
        else:
            p1 = yy
            p2 = xx
        c_lower = np.array([p2[0], p1[1]])
        c_upper = np.array([p1[0], p2[1]])
        b2 = (p1[1] - p2[1]) ** 2
    b2_over_a2 = b2 / (p2[0] - p1[0]) ** 2

    def lower(x):
        return c_lower[1] - np.sqrt(b2 - b2_over_a2 * (x - c_lower[0]) ** 2)

    def upper(x):
        return c_upper[1] + np.sqrt(b2 - b2_over_a2 * (x - c_upper[0]) ** 2)
    return {'lower': lower, 'upper': upper}