from __future__ import annotations
import bisect
from copy import copy, deepcopy
from datetime import datetime
from math import log, pi, sqrt
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MSONable
from scipy import constants
from scipy.special import comb, erfc
from pymatgen.core.structure import Structure
from pymatgen.util.due import Doi, due
def compute_sub_structure(self, sub_structure, tol: float=0.001):
    """
        Gives total Ewald energy for an sub structure in the same
        lattice. The sub_structure must be a subset of the original
        structure, with possible different charges.

        Args:
            substructure (Structure): Substructure to compute Ewald sum for.
            tol (float): Tolerance for site matching in fractional coordinates.

        Returns:
            Ewald sum of substructure.
        """
    total_energy_matrix = self.total_energy_matrix.copy()

    def find_match(site):
        for test_site in sub_structure:
            frac_diff = abs(np.array(site.frac_coords) - np.array(test_site.frac_coords)) % 1
            frac_diff = [abs(a) < tol or abs(a) > 1 - tol for a in frac_diff]
            if all(frac_diff):
                return test_site
        return None
    matches = []
    for idx, site in enumerate(self._struct):
        matching_site = find_match(site)
        if matching_site:
            new_charge = compute_average_oxidation_state(matching_site)
            old_charge = self._oxi_states[idx]
            scaling_factor = new_charge / old_charge
            matches.append(matching_site)
        else:
            scaling_factor = 0
        total_energy_matrix[idx, :] *= scaling_factor
        total_energy_matrix[:, idx] *= scaling_factor
    if len(matches) != len(sub_structure):
        output = ['Missing sites.']
        for site in sub_structure:
            if site not in matches:
                output.append(f'unmatched = {site}')
        raise ValueError('\n'.join(output))
    return sum(sum(total_energy_matrix))