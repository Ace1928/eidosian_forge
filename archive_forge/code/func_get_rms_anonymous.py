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
def get_rms_anonymous(self, struct1, struct2):
    """
        Performs an anonymous fitting, which allows distinct species in one
        structure to map to another. E.g., to compare if the Li2O and Na2O
        structures are similar.

        Args:
            struct1 (Structure): 1st structure
            struct2 (Structure): 2nd structure

        Returns:
            tuple[float, float] | tuple[None, None]: 1st element is min_rms, 2nd is min_mapping.
                min_rms is the minimum RMS distance, and min_mapping is the corresponding
                minimal species mapping that would map struct1 to struct2. (None, None) is
                returned if the minimax_rms exceeds the threshold.
        """
    struct1, struct2 = self._process_species([struct1, struct2])
    struct1, struct2, fu, s1_supercell = self._preprocess(struct1, struct2)
    matches = self._anonymous_match(struct1, struct2, fu, s1_supercell, use_rms=True, break_on_match=False)
    if matches:
        best = sorted(matches, key=lambda x: x[1][0])[0]
        return (best[1][0], best[0])
    return (None, None)