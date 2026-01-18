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
def group_structures(self, s_list, anonymous=False):
    """
        Given a list of structures, use fit to group
        them by structural equality.

        Args:
            s_list ([Structure]): List of structures to be grouped
            anonymous (bool): Whether to use anonymous mode.

        Returns:
            A list of lists of matched structures
            Assumption: if s1 == s2 but s1 != s3, than s2 and s3 will be put
            in different groups without comparison.
        """
    if self._subset:
        raise ValueError('allow_subset cannot be used with group_structures')
    original_s_list = list(s_list)
    s_list = self._process_species(s_list)
    s_list = [self._get_reduced_structure(s, self._primitive_cell, niggli=True) for s in s_list]
    if anonymous:

        def c_hash(c):
            return c.anonymized_formula
    else:
        c_hash = self._comparator.get_hash

    def s_hash(s):
        return c_hash(s[1].composition)
    sorted_s_list = sorted(enumerate(s_list), key=s_hash)
    all_groups = []
    for _, g in itertools.groupby(sorted_s_list, key=s_hash):
        unmatched = list(g)
        while len(unmatched) > 0:
            i, refs = unmatched.pop(0)
            matches = [i]
            if anonymous:
                inds = filter(lambda i: self.fit_anonymous(refs, unmatched[i][1], skip_structure_reduction=True), list(range(len(unmatched))))
            else:
                inds = filter(lambda i: self.fit(refs, unmatched[i][1], skip_structure_reduction=True), list(range(len(unmatched))))
            inds = list(inds)
            matches.extend([unmatched[i][0] for i in inds])
            unmatched = [unmatched[i] for i in range(len(unmatched)) if i not in inds]
            all_groups.append([original_s_list[i] for i in matches])
    return all_groups