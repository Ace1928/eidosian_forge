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
def generate_adsorption_structures(self, molecule, repeat=None, min_lw=5.0, translate=True, reorient=True, find_args=None):
    """Function that generates all adsorption structures for a given
        molecular adsorbate. Can take repeat argument or minimum length/width
        of precursor slab as an input.

        Args:
            molecule (Molecule): molecule corresponding to adsorbate
            repeat (3-tuple or list): repeat argument for supercell generation
            min_lw (float): minimum length and width of the slab, only used
                if repeat is None
            translate (bool): flag on whether to translate the molecule so
                that its CoM is at the origin prior to adding it to the surface
            reorient (bool): flag on whether or not to reorient adsorbate
                along the miller index
            find_args (dict): dictionary of arguments to be passed to the
                call to self.find_adsorption_sites, e.g. {"distance":2.0}
        """
    if repeat is None:
        xrep = np.ceil(min_lw / np.linalg.norm(self.slab.lattice.matrix[0]))
        yrep = np.ceil(min_lw / np.linalg.norm(self.slab.lattice.matrix[1]))
        repeat = [xrep, yrep, 1]
    structs = []
    find_args = find_args or {}
    for coords in self.find_adsorption_sites(**find_args)['all']:
        structs.append(self.add_adsorbate(molecule, coords, repeat=repeat, translate=translate, reorient=reorient))
    return structs