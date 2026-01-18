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
def get_stable_entry(self, pH, V):
    """
        Gets the stable entry at a given pH, V condition.

        Args:
            pH (float): pH at a given condition
            V (float): V at a given condition

        Returns:
            PourbaixEntry | MultiEntry: Pourbaix or multi-entry
                corresponding to the minimum energy entry at a given pH, V condition
        """
    all_gs = np.array([entry.normalized_energy_at_conditions(pH, V) for entry in self.stable_entries])
    return self.stable_entries[np.argmin(all_gs)]