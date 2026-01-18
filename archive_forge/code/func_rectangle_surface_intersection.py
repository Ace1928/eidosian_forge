from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def rectangle_surface_intersection(rectangle, f_lower, f_upper, bounds_lower=None, bounds_upper=None, check=True, numpoints_check=500):
    """
    Method to calculate the surface of the intersection of a rectangle (aligned with axes) and another surface
    defined by two functions f_lower and f_upper.

    Args:
        rectangle:
            Rectangle defined as : ((x1, x2), (y1, y2)).
        f_lower:
            Function defining the lower bound of the surface.
        f_upper:
            Function defining the upper bound of the surface.
        bounds_lower:
            Interval in which the f_lower function is defined.
        bounds_upper:
            Interval in which the f_upper function is defined.
        check:
            Whether to check if f_lower is always lower than f_upper.
        numpoints_check:
            Number of points used to check whether f_lower is always lower than f_upper

    Returns:
        The surface of the intersection of the rectangle and the surface defined by f_lower and f_upper.
    """
    x1 = np.min(rectangle[0])
    x2 = np.max(rectangle[0])
    y1 = np.min(rectangle[1])
    y2 = np.max(rectangle[1])
    if check:
        if bounds_lower is not None:
            if bounds_upper is not None:
                if not all(np.array(bounds_lower) == np.array(bounds_upper)):
                    raise ValueError('Bounds should be identical for both f_lower and f_upper')
                if '<' not in function_comparison(f1=f_lower, f2=f_upper, x1=bounds_lower[0], x2=bounds_lower[1], numpoints_check=numpoints_check):
                    raise RuntimeError('Function f_lower is not always lower or equal to function f_upper within the domain defined by the functions bounds.')
            else:
                raise ValueError('Bounds are given for f_lower but not for f_upper')
        elif bounds_upper is not None:
            if bounds_lower is None:
                raise ValueError('Bounds are given for f_upper but not for f_lower')
            if '<' not in function_comparison(f1=f_lower, f2=f_upper, x1=bounds_lower[0], x2=bounds_lower[1], numpoints_check=numpoints_check):
                raise RuntimeError('Function f_lower is not always lower or equal to function f_upper within the domain defined by the functions bounds.')
        elif '<' not in function_comparison(f1=f_lower, f2=f_upper, x1=x1, x2=x2, numpoints_check=numpoints_check):
            raise RuntimeError('Function f_lower is not always lower or equal to function f_upper within the domain defined by x1 and x2.')
    if bounds_lower is None:
        raise NotImplementedError('Bounds should be given right now ...')
    if x2 < bounds_lower[0] or x1 > bounds_lower[1]:
        return (0.0, 0.0)
    xmin = max(x1, bounds_lower[0])
    xmax = min(x2, bounds_lower[1])

    def diff(x):
        f_low_x = f_lower(x)
        f_up_x = f_upper(x)
        min_up = np.min([f_up_x, y2 * np.ones_like(f_up_x)], axis=0)
        max_lw = np.max([f_low_x, y1 * np.ones_like(f_low_x)], axis=0)
        zeros = np.zeros_like(f_up_x)
        upper = np.where(y2 >= f_low_x, np.where(y1 <= f_up_x, min_up, zeros), zeros)
        lower = np.where(y1 <= f_up_x, np.where(y2 >= f_low_x, max_lw, zeros), zeros)
        return upper - lower
    return quad(diff, xmin, xmax)