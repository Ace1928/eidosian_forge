from the a CostDB instance, for example a CSV file via CostDBCSV.
from __future__ import annotations
import abc
import csv
import itertools
import os
from collections import defaultdict
import scipy.constants as const
from monty.design_patterns import singleton
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element
from pymatgen.util.provenance import is_valid_bibtex
def get_cost_per_kg(self, comp):
    """
        Get best estimate of minimum cost/kg based on known data.

        Args:
            comp:
                Composition as a pymatgen.core.structure.Composition

        Returns:
            float of cost/kg
        """
    comp = comp if isinstance(comp, Composition) else Composition(comp)
    return self.get_cost_per_mol(comp) / (comp.weight.to('kg') * const.N_A)