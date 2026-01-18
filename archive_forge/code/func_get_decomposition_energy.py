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
def get_decomposition_energy(self, entry, pH, V):
    """
        Finds decomposition to most stable entries in eV/atom,
        supports vectorized inputs for pH and V.

        Args:
            entry (PourbaixEntry): PourbaixEntry corresponding to
                compound to find the decomposition for
            pH (float, [float]): pH at which to find the decomposition
            V (float, [float]): voltage at which to find the decomposition

        Returns:
            Decomposition energy for the entry, i. e. the energy above
                the "Pourbaix hull" in eV/atom at the given conditions
        """
    pbx_comp = Composition(self._elt_comp).fractional_composition
    entry_pbx_comp = Composition({elt: coeff for elt, coeff in entry.composition.items() if elt not in ELEMENTS_HO}).fractional_composition
    if entry_pbx_comp != pbx_comp:
        raise ValueError('Composition of stability entry does not match Pourbaix Diagram')
    entry_normalized_energy = entry.normalized_energy_at_conditions(pH, V)
    hull_energy = self.get_hull_energy(pH, V)
    decomposition_energy = entry_normalized_energy - hull_energy
    decomposition_energy /= entry.normalization_factor
    decomposition_energy /= entry.composition.num_atoms
    return decomposition_energy