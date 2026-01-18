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
def contains_peroxide(structure, relative_cutoff=1.1):
    """
    Determines if a structure contains peroxide anions.

    Args:
        structure (Structure): Input structure.
        relative_cutoff: The peroxide bond distance is 1.49 Angstrom.
            Relative_cutoff * 1.49 stipulates the maximum distance two O
            atoms must be to each other to be considered a peroxide.

    Returns:
        Boolean indicating if structure contains a peroxide anion.
    """
    return oxide_type(structure, relative_cutoff) == 'peroxide'