from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.linalg import norm
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from scipy.spatial import ConvexHull
from pymatgen.analysis.chemenv.utils.chemenv_errors import SolidAngleError
def is_anion_cation_bond(valences, ii, jj) -> bool:
    """
    Checks if two given sites are an anion and a cation.

    Args:
        valences: list of site valences
        ii: index of a site
        jj: index of another site

    Returns:
        bool: True if one site is an anion and the other is a cation (based on valences).
    """
    if valences == 'undefined':
        return True
    if valences[ii] == 0 or valences[jj] == 0:
        return True
    return valences[ii] > 0 > valences[jj] or valences[jj] > 0 > valences[ii]