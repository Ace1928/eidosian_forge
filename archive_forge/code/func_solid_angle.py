from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def solid_angle(center, coords):
    """
    Helper method to calculate the solid angle of a set of coords from the center.

    Args:
        center: Center to measure solid angle from.
        coords: List of coords to determine solid angle.

    Returns:
        The solid angle.
    """
    origin = np.array(center)
    r = [np.array(c) - origin for c in coords]
    r.append(r[0])
    n = [np.cross(r[i + 1], r[i]) for i in range(len(r) - 1)]
    n.append(np.cross(r[1], r[0]))
    phi = 0.0
    for idx in range(len(n) - 1):
        try:
            value = math.acos(-np.dot(n[idx], n[idx + 1]) / (np.linalg.norm(n[idx]) * np.linalg.norm(n[idx + 1])))
        except ValueError:
            cos = -np.dot(n[idx], n[idx + 1]) / (np.linalg.norm(n[idx]) * np.linalg.norm(n[idx + 1]))
            if 0.999999999999 < cos < 1.000000000001:
                value = math.acos(1.0)
            elif -0.999999999999 > cos > -1.000000000001:
                value = math.acos(-1.0)
            else:
                raise SolidAngleError(cos)
        phi += value
    return phi + (3 - len(r)) * math.pi