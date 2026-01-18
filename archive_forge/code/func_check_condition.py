from __future__ import annotations
from typing import TYPE_CHECKING
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import is_anion_cation_bond
from pymatgen.util.due import Doi, due
def check_condition(self, condition, structure: Structure, parameters):
    """
        Args:
            condition:
            structure:
            parameters:
        """
    if condition == self.NONE:
        return True
    if condition == self.ONLY_ACB:
        valences = parameters['valences']
        ii = parameters['site_index']
        jj = parameters['neighbor_index']
        return is_anion_cation_bond(valences, ii, jj)
    if condition == self.NO_E2SEB:
        ii = parameters['site_index']
        jj = parameters['neighbor_index']
        elems_ii = [sp.symbol for sp in structure[ii].species]
        elems_jj = [sp.symbol for sp in structure[jj].species]
        return len(set(elems_ii) & set(elems_jj)) == 0
    if condition == self.ONLY_ACB_AND_NO_E2SEB:
        valences = parameters['valences']
        ii = parameters['site_index']
        jj = parameters['neighbor_index']
        elems_ii = [sp.symbol for sp in structure[ii].species]
        elems_jj = [sp.symbol for sp in structure[jj].species]
        return len(set(elems_ii) & set(elems_jj)) == 0 and is_anion_cation_bond(valences, ii, jj)
    if condition == self.ONLY_E2OB:
        ii = parameters['site_index']
        jj = parameters['neighbor_index']
        elems_ii = [sp.symbol for sp in structure[ii].species]
        elems_jj = [sp.symbol for sp in structure[jj].species]
        return 'O' in elems_jj and 'O' not in elems_ii or ('O' in elems_ii and 'O' not in elems_jj)
    return None