from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def spline_functions(lower_points, upper_points, degree=3):
    """
    Method that creates two (upper and lower) spline functions based on points lower_points and upper_points.

    Args:
        lower_points:
            Points defining the lower function.
        upper_points:
            Points defining the upper function.
        degree:
            Degree for the spline function

    Returns:
        A dictionary with the lower and upper spline functions.
    """
    lower_xx = np.array([pp[0] for pp in lower_points])
    lower_yy = np.array([pp[1] for pp in lower_points])
    upper_xx = np.array([pp[0] for pp in upper_points])
    upper_yy = np.array([pp[1] for pp in upper_points])
    lower_spline = UnivariateSpline(lower_xx, lower_yy, k=degree, s=0)
    upper_spline = UnivariateSpline(upper_xx, upper_yy, k=degree, s=0)

    def lower(x):
        return lower_spline(x)

    def upper(x):
        return upper_spline(x)
    return {'lower': lower, 'upper': upper}