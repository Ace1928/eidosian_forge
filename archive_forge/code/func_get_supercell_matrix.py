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
def get_supercell_matrix(self, supercell, struct) -> np.ndarray | None:
    """
        Returns the matrix for transforming struct to supercell. This
        can be used for very distorted 'supercells' where the primitive cell
        is impossible to find.
        """
    if self._primitive_cell:
        raise ValueError('get_supercell_matrix cannot be used with the primitive cell option')
    struct, supercell, fu, s1_supercell = self._preprocess(struct, supercell, niggli=False)
    if not s1_supercell:
        raise ValueError('The non-supercell must be put onto the basis of the supercell, not the other way around')
    match = self._match(struct, supercell, fu, s1_supercell, use_rms=True, break_on_match=False)
    if match is None:
        return None
    return match[2]