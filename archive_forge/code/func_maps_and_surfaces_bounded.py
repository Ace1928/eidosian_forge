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
def maps_and_surfaces_bounded(self, isite, surface_calculation_options=None, additional_conditions=None):
    """
        Get the different surfaces (using boundaries) and their cn_map corresponding to the different
        distance-angle cutoffs for a given site.

        Args:
            isite: Index of the site
            surface_calculation_options: Options for the boundaries.
            additional_conditions: If additional conditions have to be considered.

        Returns:
            Surfaces and cn_map's for each distance-angle cutoff.
        """
    if self.voronoi_list2[isite] is None:
        return None
    if additional_conditions is None:
        additional_conditions = [self.AC.ONLY_ACB]
    surfaces = self.neighbors_surfaces_bounded(isite=isite, surface_calculation_options=surface_calculation_options)
    maps_and_surfaces = []
    for cn, value in self._unique_coordinated_neighbors_parameters_indices[isite].items():
        for imap, list_parameters_indices in enumerate(value):
            thissurf = 0.0
            for idp, iap, iacb in list_parameters_indices:
                if iacb in additional_conditions:
                    thissurf += surfaces[idp, iap]
            maps_and_surfaces.append({'map': (cn, imap), 'surface': thissurf, 'parameters_indices': list_parameters_indices})
    return maps_and_surfaces