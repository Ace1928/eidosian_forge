from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def changebasis(uu, vv, nn, pps):
    """
    For a list of points given in standard coordinates (in terms of e1, e2 and e3), returns the same list
    expressed in the basis (uu, vv, nn), which is supposed to be orthonormal.

    Args:
        uu: First vector of the basis
        vv: Second vector of the basis
        nn: Third vector of the basis
        pps: List of points in basis (e1, e2, e3)
    Returns:
        List of points in basis (uu, vv, nn).
    """
    MM = np.zeros([3, 3], float)
    for ii in range(3):
        MM[ii, 0] = uu[ii]
        MM[ii, 1] = vv[ii]
        MM[ii, 2] = nn[ii]
    PP = np.linalg.inv(MM)
    new_pps = []
    for pp in pps:
        new_pps.append(matrixTimesVector(PP, pp))
    return new_pps