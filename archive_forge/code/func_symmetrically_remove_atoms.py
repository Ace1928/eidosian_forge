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
def symmetrically_remove_atoms(self, indices: list[int]) -> None:
    """Remove sites from a list of indices. Will also remove the
        equivalent site on the other side of the slab to maintain symmetry.

        Args:
            indices (list[int]): The indices of the sites to remove.

        TODO(@DanielYang59):
        1. Reuse public method get_symmetric_site to get equi sites?
        2. If not 1, get_equi_sites has multiple nested loops
        """

    def get_equi_sites(slab: Slab, sites: list[int]) -> list[int]:
        """
            Get the indices of the equivalent sites of given sites.

            Parameters:
                slab (Slab): The slab structure.
                sites (list[int]): Original indices of sites.

            Returns:
                list[int]: Indices of the equivalent sites.
            """
        equi_sites = []
        for pt in sites:
            cart_point = slab.lattice.get_cartesian_coords(pt)
            dist = [site.distance_from_point(cart_point) for site in slab]
            site1 = dist.index(min(dist))
            for i, eq_sites in enumerate(slab.equivalent_sites):
                if slab[site1] in eq_sites:
                    eq_indices = slab.equivalent_indices[i]
                    break
            i1 = eq_indices[eq_sites.index(slab[site1])]
            for i2 in eq_indices:
                if i2 == i1:
                    continue
                if slab[i2].frac_coords[2] == slab[i1].frac_coords[2]:
                    continue
                slab = self.copy()
                slab.remove_sites([i1, i2])
                if slab.is_symmetric():
                    equi_sites.append(i2)
                    break
        return equi_sites
    slab_copy = SpacegroupAnalyzer(self.copy()).get_symmetrized_structure()
    sites = [slab_copy[i].frac_coords for i in indices]
    equi_sites = get_equi_sites(slab_copy, sites)
    if len(equi_sites) == len(indices):
        self.remove_sites(indices)
        self.remove_sites(equi_sites)
    else:
        warnings.warn('Equivalent sites could not be found for some indices. Surface unchanged.')