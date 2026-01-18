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
def generate_substitution_structures(self, atom, target_species=None, sub_both_sides=False, range_tol=0.01, dist_from_surf=0):
    """Function that performs substitution-type doping on the surface and
        returns all possible configurations where one dopant is substituted per
        surface. Can substitute one surface or both.

        Args:
            atom (str): atom corresponding to substitutional dopant
            sub_both_sides (bool): If true, substitute an equivalent
                site on the other surface
            target_species (list): List of specific species to substitute
            range_tol (float): Find viable substitution sites at a specific
                distance from the surface +- this tolerance
            dist_from_surf (float): Distance from the surface to find viable
                substitution sites, defaults to 0 to substitute at the surface
        """
    target_species = target_species or []
    sym_slab = SpacegroupAnalyzer(self.slab).get_symmetrized_structure()

    def substitute(site, idx):
        slab = self.slab.copy()
        props = self.slab.site_properties
        if sub_both_sides:
            eq_indices = next((indices for indices in sym_slab.equivalent_indices if idx in indices))
            for ii in eq_indices:
                if f'{sym_slab[ii].frac_coords[2]:.6f}' != f'{site.frac_coords[2]:.6f}':
                    props['surface_properties'][ii] = 'substitute'
                    slab.replace(ii, atom)
                    break
        props['surface_properties'][idx] = 'substitute'
        slab.replace(idx, atom)
        slab.add_site_property('surface_properties', props['surface_properties'])
        return slab
    substituted_slabs = []
    sorted_sites = sorted(sym_slab, key=lambda site: site.frac_coords[2])
    if sorted_sites[0].surface_properties == 'surface':
        dist = sorted_sites[0].frac_coords[2] + dist_from_surf
    else:
        dist = sorted_sites[-1].frac_coords[2] - dist_from_surf
    for idx, site in enumerate(sym_slab):
        if dist - range_tol < site.frac_coords[2] < dist + range_tol and (target_species and site.species_string in target_species or not target_species):
            substituted_slabs.append(substitute(site, idx))
    matcher = StructureMatcher()
    return [s[0] for s in matcher.group_structures(substituted_slabs)]