from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def function_comparison(f1, f2, x1, x2, numpoints_check=500):
    """
    Method that compares two functions.

    Args:
        f1: First function to compare
        f2: Second function to compare
        x1: Lower bound of the interval to compare
        x2: Upper bound of the interval to compare
        numpoints_check: Number of points used to compare the functions

    Returns:
        str: '=' if the functions are equal, '<' if f1 is always lower than f2, '>' if f1 is always larger than f2,
            f1 is always lower than or equal to f2 ("<"), f1 is always larger than or equal to f2 (">") on the
            interval [x1, x2]. If the two functions cross, a RuntimeError is thrown (i.e. we expect to compare
            functions that do not cross...)
    """
    xx = np.linspace(x1, x2, num=numpoints_check)
    y1 = f1(xx)
    y2 = f2(xx)
    if np.all(y1 < y2):
        return '<'
    if np.all(y1 > y2):
        return '>'
    if np.all(y1 == y2):
        return '='
    if np.all(y1 <= y2):
        return '<='
    if np.all(y1 >= y2):
        return '>='
    raise RuntimeError('Error in comparing functions f1 and f2 ...')