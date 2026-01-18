from __future__ import annotations
import collections
import functools
import operator
import os
from math import exp, sqrt
from typing import TYPE_CHECKING
import numpy as np
from monty.serialization import loadfn
from pymatgen.core import Element, Species, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def add_oxidation_state_by_site_fraction(structure, oxidation_states):
    """
    Add oxidation states to a structure by fractional site.

    Args:
        oxidation_states (list): List of list of oxidation states for each
            site fraction for each site.
            E.g., [[2, 4], [3], [-2], [-2], [-2]]
    """
    try:
        for idx, site in enumerate(structure):
            new_sp = collections.defaultdict(float)
            for j, (el, occu) in enumerate(get_z_ordered_elmap(site.species)):
                specie = Species(el.symbol, oxidation_states[idx][j])
                new_sp[specie] += occu
            structure[idx] = new_sp
        return structure
    except IndexError:
        raise ValueError('Oxidation state of all sites must be specified in the list.')