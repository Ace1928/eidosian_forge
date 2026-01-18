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
def near_reduce(self, coords_set, threshold=0.0001):
    """Prunes coordinate set for coordinates that are within threshold.

        Args:
            coords_set (Nx3 array-like): list or array of coordinates
            threshold (float): threshold value for distance
        """
    unique_coords = []
    coords_set = [self.slab.lattice.get_fractional_coords(coords) for coords in coords_set]
    for coord in coords_set:
        if not in_coord_list_pbc(unique_coords, coord, threshold):
            unique_coords += [coord]
    return [self.slab.lattice.get_cartesian_coords(coords) for coords in unique_coords]