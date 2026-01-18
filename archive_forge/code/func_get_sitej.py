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
def get_sitej(self, site_index, image_index):
    """
        Assuming there is some value in the connectivity array at indices
        (1, 3, 12). site_i can be obtained directly from the input structure
        (structure[1]). site_j can be obtained by passing 3, 12 to this function.

        Args:
            site_index (int): index of the site (3 in the example)
            image_index (int): index of the image (12 in the example)
        """
    atoms_n_occu = self.structure[site_index].species
    lattice = self.structure.lattice
    coords = self.structure[site_index].frac_coords + self.offsets[image_index]
    return PeriodicSite(atoms_n_occu, coords, lattice)