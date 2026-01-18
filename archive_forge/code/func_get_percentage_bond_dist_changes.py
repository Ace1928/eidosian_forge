from __future__ import annotations
import collections
import itertools
from math import acos, pi
from typing import TYPE_CHECKING
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.analysis.local_env import JmolNN, VoronoiNN
from pymatgen.core import Composition, Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_percentage_bond_dist_changes(self, max_radius: float=3.0) -> dict[int, dict[int, float]]:
    """
        Returns the percentage bond distance changes for each site up to a
        maximum radius for nearest neighbors.

        Args:
            max_radius (float): Maximum radius to search for nearest
               neighbors. This radius is applied to the initial structure,
               not the final structure.

        Returns:
            dict[int, dict[int, float]]: Bond distance changes in the form {index1: {index2: 0.011, ...}}.
                For economy of representation, the index1 is always less than index2, i.e., since bonding
                between site1 and site_n is the same as bonding between site_n and site1, there is no
                reason to duplicate the information or computation.
        """
    data: dict[int, dict[int, float]] = collections.defaultdict(dict)
    for indices in itertools.combinations(list(range(len(self.initial))), 2):
        ii, jj = sorted(indices)
        initial_dist = self.initial[ii].distance(self.initial[jj])
        if initial_dist < max_radius:
            final_dist = self.final[ii].distance(self.final[jj])
            data[ii][jj] = final_dist / initial_dist - 1
    return data