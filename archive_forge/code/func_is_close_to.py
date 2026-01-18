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
def is_close_to(self, other, rtol=0.0, atol=1e-08) -> bool:
    """
        Whether two DetailedVoronoiContainer objects are close to each other.

        Args:
            other: Another DetailedVoronoiContainer to be compared with.
            rtol: Relative tolerance to compare values.
            atol: Absolute tolerance to compare values.

        Returns:
            bool: True if the two DetailedVoronoiContainer are close to each other.
        """
    isclose = np.isclose(self.normalized_angle_tolerance, other.normalized_angle_tolerance, rtol=rtol, atol=atol) and np.isclose(self.normalized_distance_tolerance, other.normalized_distance_tolerance, rtol=rtol, atol=atol) and (self.additional_conditions == other.additional_conditions) and (self.valences == other.valences)
    if not isclose:
        return isclose
    for isite, site_voronoi in enumerate(self.voronoi_list2):
        self_to_other_nbs = {}
        for inb, nb in enumerate(site_voronoi):
            if nb is None:
                if other.voronoi_list2[isite] is None:
                    continue
                return False
            if other.voronoi_list2[isite] is None:
                return False
            nb_other = None
            for inb2, nb2 in enumerate(other.voronoi_list2[isite]):
                if nb['site'] == nb2['site']:
                    self_to_other_nbs[inb] = inb2
                    nb_other = nb2
                    break
            if nb_other is None:
                return False
            if not np.isclose(nb['distance'], nb_other['distance'], rtol=rtol, atol=atol):
                return False
            if not np.isclose(nb['angle'], nb_other['angle'], rtol=rtol, atol=atol):
                return False
            if not np.isclose(nb['normalized_distance'], nb_other['normalized_distance'], rtol=rtol, atol=atol):
                return False
            if not np.isclose(nb['normalized_angle'], nb_other['normalized_angle'], rtol=rtol, atol=atol):
                return False
            if nb['index'] != nb_other['index']:
                return False
            if nb['site'] != nb_other['site']:
                return False
    return True