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
def move_to_other_side(self, init_slab: Slab, index_of_sites: list[int]) -> Slab:
    """Move surface sites to the opposite surface of the Slab.

        If a selected site resides on the top half of the Slab,
        it would be moved to the bottom side, and vice versa.
        The distance moved is equal to the thickness of the Slab.

        Note:
            You should only use this method on sites close to the
            surface, otherwise it would end up deep inside the
            vacuum layer.

        Args:
            init_slab (Slab): The Slab whose sites would be moved.
            index_of_sites (list[int]): Indices representing
                the sites to move.

        Returns:
            Slab: The Slab with selected sites moved.
        """
    height: float = self._proj_height
    if self.in_unit_planes:
        height /= self.parent.lattice.d_hkl(self.miller_index)
    n_layers_slab: int = math.ceil(self.min_slab_size / height)
    n_layers_vac: int = math.ceil(self.min_vac_size / height)
    n_layers: int = n_layers_slab + n_layers_vac
    frac_dist: float = n_layers_slab / n_layers
    top_site_index: list[int] = []
    bottom_site_index: list[int] = []
    for idx in index_of_sites:
        if init_slab[idx].frac_coords[2] >= init_slab.center_of_mass[2]:
            top_site_index.append(idx)
        else:
            bottom_site_index.append(idx)
    slab = init_slab.copy()
    slab.translate_sites(top_site_index, vector=[0, 0, -frac_dist], frac_coords=True)
    slab.translate_sites(bottom_site_index, vector=[0, 0, frac_dist], frac_coords=True)
    return Slab(init_slab.lattice, slab.species, slab.frac_coords, init_slab.miller_index, init_slab.oriented_unit_cell, init_slab.shift, init_slab.scale_factor, energy=init_slab.energy)