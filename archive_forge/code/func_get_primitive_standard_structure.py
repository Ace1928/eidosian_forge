from __future__ import annotations
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import cos, sin
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import scipy.cluster
import spglib
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list, pbc_diff
from pymatgen.util.due import Doi, due
@cite_conventional_cell_algo
def get_primitive_standard_structure(self, international_monoclinic=True, keep_site_properties=False):
    """Gives a structure with a primitive cell according to certain standards. The
        standards are defined in Setyawan, W., & Curtarolo, S. (2010). High-throughput
        electronic band structure calculations: Challenges and tools. Computational
        Materials Science, 49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.

        Args:
            international_monoclinic (bool): Whether to convert to proper international convention
                such that beta is the non-right angle.
            keep_site_properties (bool): Whether to keep the input site properties (including
                magnetic moments) on the sites that are still present after the refinement. Note:
                This is disabled by default because the magnetic moments are not always directly
                transferable between unit cell definitions. For instance, long-range magnetic
                ordering or antiferromagnetic character may no longer be present (or exist in
                the same way) in the returned structure. If keep_site_properties is True,
                each site retains the same site property as in the original structure without
                further adjustment.

        Returns:
            The structure in a primitive standardized cell
        """
    conv = self.get_conventional_standard_structure(international_monoclinic=international_monoclinic, keep_site_properties=keep_site_properties)
    lattice = self.get_lattice_type()
    if 'P' in self.get_space_group_symbol() or lattice == 'hexagonal':
        return conv
    transf = self.get_conventional_to_primitive_transformation_matrix(international_monoclinic=international_monoclinic)
    new_sites = []
    lattice = Lattice(np.dot(transf, conv.lattice.matrix))
    for site in conv:
        new_s = PeriodicSite(site.specie, site.coords, lattice, to_unit_cell=True, coords_are_cartesian=True, properties=site.properties)
        if not any(map(new_s.is_periodic_image, new_sites)):
            new_sites.append(new_s)
    if lattice == 'rhombohedral':
        prim = Structure.from_sites(new_sites)
        lengths = prim.lattice.lengths
        angles = prim.lattice.angles
        a = lengths[0]
        alpha = math.pi * angles[0] / 180
        new_matrix = [[a * cos(alpha / 2), -a * sin(alpha / 2), 0], [a * cos(alpha / 2), a * sin(alpha / 2), 0], [a * cos(alpha) / cos(alpha / 2), 0, a * math.sqrt(1 - cos(alpha) ** 2 / cos(alpha / 2) ** 2)]]
        new_sites = []
        lattice = Lattice(new_matrix)
        for site in prim:
            new_s = PeriodicSite(site.specie, site.frac_coords, lattice, to_unit_cell=True, properties=site.properties)
            if not any(map(new_s.is_periodic_image, new_sites)):
                new_sites.append(new_s)
        return Structure.from_sites(new_sites)
    return Structure.from_sites(new_sites)