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
def assign_selective_dynamics(cls, slab):
    """Helper function to assign selective dynamics site_properties based
        on surface, subsurface site properties.

        Args:
            slab (Slab): slab for which to assign selective dynamics
        """
    sd_list = []
    sd_list = [[False, False, False] if site.properties['surface_properties'] == 'subsurface' else [True, True, True] for site in slab]
    new_sp = slab.site_properties
    new_sp['selective_dynamics'] = sd_list
    return slab.copy(site_properties=new_sp)