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
def assign_site_properties(self, slab: Slab, height=0.9):
    """Assigns site properties."""
    if 'surface_properties' in slab.site_properties:
        return slab
    surf_sites = self.find_surface_sites_by_height(slab, height)
    surf_props = ['surface' if site in surf_sites else 'subsurface' for site in slab]
    return slab.copy(site_properties={'surface_properties': surf_props})