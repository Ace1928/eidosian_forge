from __future__ import annotations
import warnings
import numpy as np
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.core import Element, Species
def get_dopants_from_substitution_probabilities(structure, num_dopants=5, threshold=0.001, match_oxi_sign=False):
    """
    Get dopant suggestions based on substitution probabilities.

    Args:
        structure (Structure): A pymatgen structure decorated with
            oxidation states.
        num_dopants (int): The number of suggestions to return for
            n- and p-type dopants.
        threshold (float): Probability threshold for substitutions.
        match_oxi_sign (bool): Whether to force the dopant and original species
            to have the same sign of oxidation state. E.g. If the original site
            is in a negative charge state, then only negative dopants will be
            returned.

    Returns:
        dict: Dopant suggestions, given as a dictionary with keys "n_type" and
            "p_type". The suggestions for each doping type are given as a list of
            dictionaries, each with they keys:

            - "probability": The probability of substitution.
            - "dopant_species": The dopant species.
            - "original_species": The substituted species.
    """
    els_have_oxi_states = [hasattr(s, 'oxi_state') for s in structure.species]
    if not all(els_have_oxi_states):
        raise ValueError('All sites in structure must have oxidation states to predict dopants.')
    sp = SubstitutionPredictor(threshold=threshold)
    subs = [sp.list_prediction([s]) for s in set(structure.species)]
    subs = [{'probability': pred['probability'], 'dopant_species': next(iter(pred['substitutions'])), 'original_species': next(iter(pred['substitutions'].values()))} for species_preds in subs for pred in species_preds]
    subs.sort(key=lambda x: x['probability'], reverse=True)
    return _get_dopants(subs, num_dopants, match_oxi_sign)