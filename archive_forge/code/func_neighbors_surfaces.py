from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.json import MSONable
from scipy.spatial import Voronoi
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import (
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.analysis.chemenv.utils.math_utils import normal_cdf_step
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
def neighbors_surfaces(self, isite, surface_calculation_type=None, max_dist=2.0):
    """
        Get the different surfaces corresponding to the different distance-angle cutoffs for a given site.

        Args:
            isite: Index of the site
            surface_calculation_type: How to compute the surface.
            max_dist: The maximum distance factor to be considered.

        Returns:
            Surfaces for each distance-angle cutoff.
        """
    if self.voronoi_list2[isite] is None:
        return None
    bounds_and_limits = self.voronoi_parameters_bounds_and_limits(isite, surface_calculation_type, max_dist)
    distance_bounds = bounds_and_limits['distance_bounds']
    angle_bounds = bounds_and_limits['angle_bounds']
    surfaces = np.zeros((len(distance_bounds), len(angle_bounds)), float)
    for idp in range(len(distance_bounds) - 1):
        this_dist_plateau = distance_bounds[idp + 1] - distance_bounds[idp]
        for iap in range(len(angle_bounds) - 1):
            this_ang_plateau = angle_bounds[iap + 1] - angle_bounds[iap]
            surfaces[idp][iap] = np.absolute(this_dist_plateau * this_ang_plateau)
    return surfaces