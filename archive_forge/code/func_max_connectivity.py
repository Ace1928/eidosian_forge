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
@property
def max_connectivity(self):
    """
        Returns the 2d array [site_i, site_j] that represents the maximum connectivity of
        site i to any periodic image of site j.
        """
    return np.max(self.connectivity_array, axis=2)