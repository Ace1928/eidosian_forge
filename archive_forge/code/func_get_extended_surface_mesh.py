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
def get_extended_surface_mesh(self, repeat=(5, 5, 1)):
    """Gets an extended surface mesh for to use for adsorption site finding
        by constructing supercell of surface sites.

        Args:
            repeat (3-tuple): repeat for getting extended surface mesh
        """
    surf_str = Structure.from_sites(self.surface_sites)
    surf_str.make_supercell(repeat)
    return surf_str