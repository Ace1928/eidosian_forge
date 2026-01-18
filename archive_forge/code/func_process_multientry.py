from __future__ import annotations
import itertools
import logging
import re
import warnings
from copy import deepcopy
from functools import cmp_to_key, partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, no_type_check
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.special import comb
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import Composition, Element
from pymatgen.core.ion import Ion
from pymatgen.entries.compatibility import MU_H2O
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.util.coord import Simplex
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import Stringify
@staticmethod
def process_multientry(entry_list, prod_comp, coeff_threshold=0.0001):
    """
        Static method for finding a multientry based on
        a list of entries and a product composition.
        Essentially checks to see if a valid aqueous
        reaction exists between the entries and the
        product composition and returns a MultiEntry
        with weights according to the coefficients if so.

        Args:
            entry_list ([Entry]): list of entries from which to
                create a MultiEntry
            prod_comp (Composition): composition constraint for setting
                weights of MultiEntry
            coeff_threshold (float): threshold of stoichiometric
                coefficients to filter, if weights are lower than
                this value, the entry is not returned
        """
    dummy_oh = [Composition('H'), Composition('O')]
    try:
        entry_comps = [entry.composition for entry in entry_list]
        rxn = Reaction(entry_comps + dummy_oh, [prod_comp])
        react_coeffs = [-coeff for coeff in rxn.coeffs[:len(entry_list)]]
        all_coeffs = [*react_coeffs, rxn.get_coeff(prod_comp)]
        if all((coeff > coeff_threshold for coeff in all_coeffs)):
            return MultiEntry(entry_list, weights=react_coeffs)
        return None
    except ReactionError:
        return None