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
def neighbors_surfaces_bounded(self, isite, surface_calculation_options=None):
    """
        Get the different surfaces (using boundaries) corresponding to the different distance-angle cutoffs
        for a given site.

        Args:
            isite: Index of the site.
            surface_calculation_options: Options for the boundaries.

        Returns:
            Surfaces for each distance-angle cutoff.
        """
    if self.voronoi_list2[isite] is None:
        return None
    if surface_calculation_options is None:
        surface_calculation_options = {'type': 'standard_elliptic', 'distance_bounds': {'lower': 1.2, 'upper': 1.8}, 'angle_bounds': {'lower': 0.1, 'upper': 0.8}}
    if surface_calculation_options['type'] in ['standard_elliptic', 'standard_diamond', 'standard_spline']:
        plot_type = {'distance_parameter': ('initial_normalized', None), 'angle_parameter': ('initial_normalized', None)}
    else:
        raise ValueError(f'Type {surface_calculation_options['type']!r} for the surface calculation in DetailedVoronoiContainer is invalid')
    max_dist = surface_calculation_options['distance_bounds']['upper'] + 0.1
    bounds_and_limits = self.voronoi_parameters_bounds_and_limits(isite=isite, plot_type=plot_type, max_dist=max_dist)
    distance_bounds = bounds_and_limits['distance_bounds']
    angle_bounds = bounds_and_limits['angle_bounds']
    lower_and_upper_functions = get_lower_and_upper_f(surface_calculation_options=surface_calculation_options)
    mindist = surface_calculation_options['distance_bounds']['lower']
    maxdist = surface_calculation_options['distance_bounds']['upper']
    minang = surface_calculation_options['angle_bounds']['lower']
    maxang = surface_calculation_options['angle_bounds']['upper']
    f_lower = lower_and_upper_functions['lower']
    f_upper = lower_and_upper_functions['upper']
    surfaces = np.zeros((len(distance_bounds), len(angle_bounds)), float)
    for idp in range(len(distance_bounds) - 1):
        dp1 = distance_bounds[idp]
        dp2 = distance_bounds[idp + 1]
        if dp2 < mindist or dp1 > maxdist:
            continue
        d1 = max(dp1, mindist)
        d2 = min(dp2, maxdist)
        for iap in range(len(angle_bounds) - 1):
            ap1 = angle_bounds[iap]
            ap2 = angle_bounds[iap + 1]
            if ap1 > ap2:
                ap1 = angle_bounds[iap + 1]
                ap2 = angle_bounds[iap]
            if ap2 < minang or ap1 > maxang:
                continue
            intersection, _interror = rectangle_surface_intersection(rectangle=((d1, d2), (ap1, ap2)), f_lower=f_lower, f_upper=f_upper, bounds_lower=[mindist, maxdist], bounds_upper=[mindist, maxdist], check=False)
            surfaces[idp][iap] = intersection
    return surfaces