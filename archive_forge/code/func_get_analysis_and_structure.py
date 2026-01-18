from __future__ import annotations
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal, cast
import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import LocalStructOrderParams, get_neighbors_of_site_with_index
from pymatgen.core import Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_analysis_and_structure(self, structure: Structure, calculate_valences: bool=True, guesstimate_spin: bool=False, op_threshold: float=0.1) -> tuple[dict, Structure]:
    """Obtain an analysis of a given structure and if it may be Jahn-Teller
        active or not. This is a heuristic, and may give false positives and
        false negatives (false positives are preferred).

        Args:
            structure: input structure
            calculate_valences: whether to attempt to calculate valences or not, structure
                should have oxidation states to perform analysis (Default value = True)
            guesstimate_spin: whether to guesstimate spin state from magnetic moments
                or not, use with caution (Default value = False)
            op_threshold: threshold for order parameter above which to consider site
                to match an octahedral or tetrahedral motif, since Jahn-Teller structures
                can often be
                quite distorted, this threshold is smaller than one might expect

        Returns:
            analysis of structure, with key 'strength' which may be 'none', 'strong',
            'weak', or 'unknown' (Default value = 0.1) and decorated structure
        """
    structure = structure.get_primitive_structure()
    if calculate_valences:
        bva = BVAnalyzer()
        structure = bva.get_oxi_state_decorated_structure(structure)
    symmetrized_structure = SpacegroupAnalyzer(structure).get_symmetrized_structure()
    op = LocalStructOrderParams(['oct', 'tet'])
    jt_sites = []
    non_jt_sites = []
    for indices in symmetrized_structure.equivalent_indices:
        idx = indices[0]
        site = symmetrized_structure[idx]
        if isinstance(site.specie, Species) and site.specie.element.is_transition_metal:
            order_params = op.get_order_parameters(symmetrized_structure, idx)
            if order_params[0] > order_params[1] and order_params[0] > op_threshold:
                motif = 'oct'
                motif_order_parameter = order_params[0]
            elif order_params[1] > op_threshold:
                motif = 'tet'
                motif_order_parameter = order_params[1]
            else:
                motif = 'unknown'
                motif_order_parameter = None
            if motif in ['oct', 'tet']:
                motif = cast(Literal['oct', 'tet'], motif)
                if guesstimate_spin and 'magmom' in site.properties:
                    magmom = site.properties['magmom']
                    spin_state = self._estimate_spin_state(site.specie, motif, magmom)
                else:
                    spin_state = 'unknown'
                magnitude = self.get_magnitude_of_effect_from_species(site.specie, spin_state, motif)
                if magnitude != 'none':
                    ligands = get_neighbors_of_site_with_index(structure, idx, approach='min_dist', delta=0.15)
                    ligand_bond_lengths = [ligand.distance(structure[idx]) for ligand in ligands]
                    ligands_species = list({str(ligand.specie) for ligand in ligands})
                    ligand_bond_length_spread = max(ligand_bond_lengths) - min(ligand_bond_lengths)

                    def trim(f):
                        """Avoid storing to unreasonable precision, hurts readability."""
                        return float(f'{f:.4f}')
                    if len(ligands_species) == 1:
                        jt_sites.append({'strength': magnitude, 'motif': motif, 'motif_order_parameter': trim(motif_order_parameter), 'spin_state': spin_state, 'species': str(site.specie), 'ligand': ligands_species[0], 'ligand_bond_lengths': [trim(length) for length in ligand_bond_lengths], 'ligand_bond_length_spread': trim(ligand_bond_length_spread), 'site_indices': indices})
                else:
                    non_jt_sites.append({'site_indices': indices, 'strength': 'none', 'reason': 'Not Jahn-Teller active for this electronic configuration.'})
            else:
                non_jt_sites.append({'site_indices': indices, 'strength': 'none', 'reason': f'motif={motif!r}'})
    if jt_sites:
        analysis: dict[str, Any] = {'active': True}
        strong_magnitudes = [site['strength'] == 'strong' for site in jt_sites]
        if any(strong_magnitudes):
            analysis['strength'] = 'strong'
        else:
            analysis['strength'] = 'weak'
        analysis['sites'] = jt_sites
        return (analysis, structure)
    return ({'active': False, 'sites': non_jt_sites}, structure)