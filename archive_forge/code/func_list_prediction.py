from __future__ import annotations
import functools
import itertools
import json
import logging
import math
import os
from collections import defaultdict
from operator import mul
from typing import TYPE_CHECKING
from monty.design_patterns import cached_class
from pymatgen.core import Species, get_el_sp
from pymatgen.util.due import Doi, due
def list_prediction(self, species, to_this_composition=True):
    """
        Args:
            species:
                list of species
            to_this_composition:
                If true, substitutions with this as a final composition
                will be found. If false, substitutions with this as a
                starting composition will be found (these are slightly
                different).

        Returns:
            List of predictions in the form of dictionaries.
            If to_this_composition is true, the values of the dictionary
            will be from the list species. If false, the keys will be
            from that list.
        """
    for sp in species:
        if get_el_sp(sp) not in self.p.species:
            raise ValueError(f'the species {sp} is not allowed for the probability model you are using')
    max_probabilities = []
    for s1 in species:
        if to_this_composition:
            max_p = max((self.p.cond_prob(s2, s1) for s2 in self.p.species))
        else:
            max_p = max((self.p.cond_prob(s1, s2) for s2 in self.p.species))
        max_probabilities.append(max_p)
    output = []

    def _recurse(output_prob, output_species):
        best_case_prob = list(max_probabilities)
        best_case_prob[:len(output_prob)] = output_prob
        if functools.reduce(mul, best_case_prob) > self.threshold:
            if len(output_species) == len(species):
                odict = {'probability': functools.reduce(mul, best_case_prob)}
                if to_this_composition:
                    odict['substitutions'] = dict(zip(output_species, species))
                else:
                    odict['substitutions'] = dict(zip(species, output_species))
                if len(output_species) == len(set(output_species)):
                    output.append(odict)
                return
            for sp in self.p.species:
                i = len(output_prob)
                prob = self.p.cond_prob(sp, species[i]) if to_this_composition else self.p.cond_prob(species[i], sp)
                _recurse([*output_prob, prob], [*output_species, sp])
    _recurse([], [])
    logging.info(f'{len(output)} substitutions found')
    return output