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
def get_best_electronegativity_anonymous_mapping(self, struct1: Structure, struct2: Structure) -> dict | None:
    """
        Performs an anonymous fitting, which allows distinct species in one
        structure to map to another. E.g., to compare if the Li2O and Na2O
        structures are similar. If multiple substitutions are within tolerance
        this will return the one which minimizes the difference in
        electronegativity between the matches species.

        Args:
            struct1 (Structure): 1st structure
            struct2 (Structure): 2nd structure

        Returns:
            min_mapping (dict): Mapping of struct1 species to struct2 species
        """
    struct1, struct2 = self._process_species([struct1, struct2])
    struct1, struct2, fu, s1_supercell = self._preprocess(struct1, struct2)
    matches = self._anonymous_match(struct1, struct2, fu, s1_supercell, use_rms=True, break_on_match=True)
    if matches:
        min_X_diff = np.inf
        for match in matches:
            X_diff = 0
            for key, val in match[0].items():
                X_diff += struct1.composition[key] * (key.X - val.X) ** 2
            if X_diff < min_X_diff:
                min_X_diff = X_diff
                best = match[0]
        return best
    return None