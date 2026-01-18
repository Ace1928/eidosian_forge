from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def diamond_functions(xx, yy, y_x0, x_y0):
    """
    Method that creates two upper and lower functions based on points xx and yy
    as well as intercepts defined by y_x0 and x_y0. The resulting functions
    form kind of a distorted diamond-like structure aligned from
    point xx to point yy.

    Schematically :

    xx is symbolized by x, yy is symbolized by y, y_x0 is equal to the distance
    from x to a, x_y0 is equal to the distance from x to b, the lines a-p and
    b-q are parallel to the line x-y such that points p and q are
    obtained automatically.
    In case of an increasing diamond the lower function is x-b-q and the upper
    function is a-p-y while in case of a
    decreasing diamond, the lower function is a-p-y and the upper function is
    x-b-q.

           Increasing diamond      |     Decreasing diamond
                     p--y                    x----b
                    /  /|                    |\\    \\
                   /  / |                    | \\    q
                  /  /  |                    a  \\   |
                 a  /   |                     \\  \\  |
                 | /    q                      \\  \\ |
                 |/    /                        \\  \\|
                 x----b                          p--y

    Args:
        xx:
            First point
        yy:
            Second point

    Returns:
        A dictionary with the lower and upper diamond functions.
    """
    np_xx = np.array(xx)
    np_yy = np.array(yy)
    if np.any(np_xx == np_yy):
        raise RuntimeError('Invalid points for diamond_functions')
    if np.all(np_xx < np_yy) or np.all(np_xx > np_yy):
        if np_xx[0] < np_yy[0]:
            p1 = np_xx
            p2 = np_yy
        else:
            p1 = np_yy
            p2 = np_xx
    elif np_xx[0] < np_yy[0]:
        p1 = np_xx
        p2 = np_yy
    else:
        p1 = np_yy
        p2 = np_xx
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    if slope > 0.0:
        x_bpoint = p1[0] + x_y0
        myy = p1[1]
        bq_intercept = myy - slope * x_bpoint
        myx = p1[0]
        myy = p1[1] + y_x0
        ap_intercept = myy - slope * myx
        x_ppoint = (p2[1] - ap_intercept) / slope

        def lower(x):
            return np.where(x <= x_bpoint, p1[1] * np.ones_like(x), slope * x + bq_intercept)

        def upper(x):
            return np.where(x >= x_ppoint, p2[1] * np.ones_like(x), slope * x + ap_intercept)
    else:
        x_bpoint = p1[0] + x_y0
        myy = p1[1]
        bq_intercept = myy - slope * x_bpoint
        myx = p1[0]
        myy = p1[1] - y_x0
        ap_intercept = myy - slope * myx
        x_ppoint = (p2[1] - ap_intercept) / slope

        def lower(x):
            return np.where(x >= x_ppoint, p2[1] * np.ones_like(x), slope * x + ap_intercept)

        def upper(x):
            return np.where(x <= x_bpoint, p1[1] * np.ones_like(x), slope * x + bq_intercept)
    return {'lower': lower, 'upper': upper}