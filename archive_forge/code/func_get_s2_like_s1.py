from __future__ import annotations
import abc
import itertools
from typing import TYPE_CHECKING, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core import Composition, Lattice, Structure, get_el_sp
from pymatgen.optimization.linear_assignment import LinearAssignment
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.util.coord_cython import is_coord_subset_pbc, pbc_shortest_vectors
def get_s2_like_s1(self, struct1, struct2, include_ignored_species=True):
    """
        Performs transformations on struct2 to put it in a basis similar to
        struct1 (without changing any of the inter-site distances).

        Args:
            struct1 (Structure): Reference structure
            struct2 (Structure): Structure to transform.
            include_ignored_species (bool): Defaults to True,
                the ignored_species is also transformed to the struct1
                lattice orientation, though obviously there is no direct
                matching to existing sites.

        Returns:
            A structure object similar to struct1, obtained by making a
            supercell, sorting, and translating struct2.
        """
    s1, s2 = self._process_species([struct1, struct2])
    trans = self.get_transformation(s1, s2)
    if trans is None:
        return None
    sc, t, mapping = trans
    sites = list(s2)
    sites.extend([site for site in struct2 if site not in s2])
    temp = Structure.from_sites(sites)
    temp.make_supercell(sc)
    temp.translate_sites(list(range(len(temp))), t)
    for ii, jj in enumerate(mapping[:len(s1)]):
        if jj is not None:
            vec = np.round(struct1[ii].frac_coords - temp[jj].frac_coords)
            temp.translate_sites(jj, vec, to_unit_cell=False)
    sites = [temp.sites[i] for i in mapping if i is not None]
    if include_ignored_species:
        start = int(round(len(temp) / len(struct2) * len(s2)))
        sites.extend(temp.sites[start:])
    return Structure.from_sites(sites)