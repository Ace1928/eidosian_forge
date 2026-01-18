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
def get_z_ranges(bonds: dict[tuple[Species | Element, Species | Element], float]) -> list[tuple[float, float]]:
    """Collect occupied z ranges where each z_range is a (lower_z, upper_z) tuple.

            This method examines all sites in the oriented unit cell (OUC)
            and considers all neighboring sites within the specified bond distance
            for each site. If a site and its neighbor meet bonding and species
            requirements, their respective z-ranges will be collected.

            Args:
                bonds (dict): A {(species1, species2): max_bond_dist} dict.
                tol (float): Fractional tolerance for determine overlapping positions.
            """
    bonds = {(get_el_sp(s1), get_el_sp(s2)): dist for (s1, s2), dist in bonds.items()}
    z_ranges = []
    for (sp1, sp2), bond_dist in bonds.items():
        for site in self.oriented_unit_cell:
            if sp1 in site.species:
                for nn in self.oriented_unit_cell.get_neighbors(site, bond_dist):
                    if sp2 in nn.species:
                        z_range = tuple(sorted([site.frac_coords[2], nn.frac_coords[2]]))
                        if z_range[1] > 1:
                            z_ranges.extend([(z_range[0], 1), (0, z_range[1] - 1)])
                        elif z_range[0] < 0:
                            z_ranges.extend([(0, z_range[1]), (z_range[0] + 1, 1)])
                        elif z_range[0] != z_range[1]:
                            z_ranges.append(z_range)
    return z_ranges