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
def get_slabs(self, bonds: dict[tuple[Species | Element, Species | Element], float] | None=None, ftol: float=0.1, tol: float=0.1, max_broken_bonds: int=0, symmetrize: bool=False, repair: bool=False) -> list[Slab]:
    """Generate slabs with shift values calculated from the internal
        calculate_possible_shifts method. If the user decide to avoid breaking
        any polyhedral bond (by setting `bonds`), any shift value that do so
        would be filtered out.

        Args:
            bonds (dict): A {(species1, species2): max_bond_dist} dict.
                For example, PO4 groups may be defined as {("P", "O"): 3}.
            tol (float): Fractional tolerance for getting primitive cells
                and matching structures.
            ftol (float): Threshold for fcluster to check if two atoms are
                on the same plane. Default to 0.1 Angstrom in the direction of
                the surface normal.
            max_broken_bonds (int): Maximum number of allowable broken bonds
                for the slab. Use this to limit number of slabs. Defaults to 0,
                which means no bonds could be broken.
            symmetrize (bool): Whether to enforce the equivalency of slab surfaces.
            repair (bool): Whether to repair terminations with broken bonds (True)
                or just omit them (False). Default to False as repairing terminations
                can lead to many more possible slabs.

        Returns:
            list[Slab]: All possible Slabs of a particular surface,
                sorted by the number of bonds broken.
        """

    def gen_possible_shifts(ftol: float) -> list[float]:
        """Generate possible shifts by clustering z coordinates.

            Args:
                ftol (float): Threshold for fcluster to check if
                    two atoms are on the same plane.
            """
        frac_coords = self.oriented_unit_cell.frac_coords
        n_atoms = len(frac_coords)
        if n_atoms == 1:
            shift = frac_coords[0][2] + 0.5
            return [shift - math.floor(shift)]
        dist_matrix = np.zeros((n_atoms, n_atoms))
        for i, j in itertools.combinations(list(range(n_atoms)), 2):
            if i != j:
                z_dist = frac_coords[i][2] - frac_coords[j][2]
                z_dist = abs(z_dist - round(z_dist)) * self._proj_height
                dist_matrix[i, j] = z_dist
                dist_matrix[j, i] = z_dist
        z_matrix = linkage(squareform(dist_matrix))
        clusters = fcluster(z_matrix, ftol, criterion='distance')
        clst_loc = {c: frac_coords[i][2] for i, c in enumerate(clusters)}
        possible_clst = [coord - math.floor(coord) for coord in sorted(clst_loc.values())]
        n_shifts = len(possible_clst)
        shifts = []
        for i in range(n_shifts):
            if i == n_shifts - 1:
                shift = (possible_clst[0] + 1 + possible_clst[i]) * 0.5
            else:
                shift = (possible_clst[i] + possible_clst[i + 1]) * 0.5
            shifts.append(shift - math.floor(shift))
        return sorted(shifts)

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
    z_ranges = [] if bonds is None else get_z_ranges(bonds)
    slabs = []
    for shift in gen_possible_shifts(ftol=ftol):
        bonds_broken = 0
        for z_range in z_ranges:
            if z_range[0] <= shift <= z_range[1]:
                bonds_broken += 1
        slab = self.get_slab(shift=shift, tol=tol, energy=bonds_broken)
        if bonds_broken <= max_broken_bonds:
            slabs.append(slab)
        elif repair and bonds is not None:
            slabs.append(self.repair_broken_bonds(slab=slab, bonds=bonds))
    matcher = StructureMatcher(ltol=tol, stol=tol, primitive_cell=False, scale=False)
    final_slabs = []
    for group in matcher.group_structures(slabs):
        if symmetrize:
            sym_slabs = self.nonstoichiometric_symmetrized_slab(group[0])
            final_slabs.extend(sym_slabs)
        else:
            final_slabs.append(group[0])
    if symmetrize:
        matcher_sym = StructureMatcher(ltol=tol, stol=tol, primitive_cell=False, scale=False)
        final_slabs = [group[0] for group in matcher_sym.group_structures(final_slabs)]
    return sorted(final_slabs, key=lambda slab: slab.energy)