from __future__ import annotations
import copy
import itertools
import json
import logging
import math
import os
import warnings
from functools import reduce
from math import gcd, isclose
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.fractions import lcm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, PeriodicSite, Structure, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.util.due import Doi, due
def build_slabs(self) -> list[Slab]:
    """Build reconstructed Slabs by:
            (1) Obtaining the unreconstructed Slab using the specified
                parameters for the SlabGenerator.
            (2) Applying the appropriate lattice transformation to the
                a and b lattice vectors.
            (3) Remove and then add specified sites from both surfaces.

        Returns:
            list[Slab]: The reconstructed slabs.
        """
    slabs = self.get_unreconstructed_slabs()
    recon_slabs = []
    for slab in slabs:
        z_spacing = get_d(slab)
        top_site = sorted(slab, key=lambda site: site.frac_coords[2])[-1].coords
        if 'points_to_remove' in self.reconstruction_json:
            sites_to_rm: list = copy.deepcopy(self.reconstruction_json['points_to_remove'])
            for site in sites_to_rm:
                site[2] = slab.lattice.get_fractional_coords([top_site[0], top_site[1], top_site[2] + site[2] * z_spacing])[2]
                cart_point = slab.lattice.get_cartesian_coords(site)
                distances: list[float] = [site.distance_from_point(cart_point) for site in slab]
                nearest_site = distances.index(min(distances))
                slab.symmetrically_remove_atoms(indices=[nearest_site])
        if 'points_to_add' in self.reconstruction_json:
            sites_to_add: list = copy.deepcopy(self.reconstruction_json['points_to_add'])
            for site in sites_to_add:
                site[2] = slab.lattice.get_fractional_coords([top_site[0], top_site[1], top_site[2] + site[2] * z_spacing])[2]
                slab.symmetrically_add_atom(species=slab[0].specie, point=site)
        slab.reconstruction = self.name
        slab.recon_trans_matrix = self.trans_matrix
        ouc = slab.oriented_unit_cell.copy()
        ouc.make_supercell(self.trans_matrix)
        slab.oriented_unit_cell = ouc
        recon_slabs.append(slab)
    return recon_slabs