from __future__ import annotations
import itertools
import os
from typing import TYPE_CHECKING
import numpy as np
from matplotlib import patches
from matplotlib.path import Path
from monty.serialization import loadfn
from scipy.spatial import Delaunay
from pymatgen import vis
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.core.surface import generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list_pbc
@classmethod
def ensemble_center(cls, site_list, indices, cartesian=True):
    """Finds the center of an ensemble of sites selected from a list of
        sites. Helper method for the find_adsorption_sites algorithm.

        Args:
            site_list (list of sites): list of sites
            indices (list of ints): list of ints from which to select
                sites from site list
            cartesian (bool): whether to get average fractional or
                Cartesian coordinate
        """
    if cartesian:
        return np.average([site_list[idx].coords for idx in indices], axis=0)
    return np.average([site_list[idx].frac_coords for idx in indices], axis=0)