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
def get_lowest_decomposition(self, composition):
    """
        Get the decomposition leading to lowest cost.

        Args:
            composition:
                Composition as a pymatgen.core.structure.Composition

        Returns:
            Decomposition as a dict of {Entry: amount}
        """
    entries_list = []
    elements = [e.symbol for e in composition.elements]
    for idx in range(len(elements)):
        for combi in itertools.combinations(elements, idx + 1):
            chemsys = [Element(e) for e in combi]
            x = self.costdb.get_entries(chemsys)
            entries_list.extend(x)
    try:
        pd = PhaseDiagram(entries_list)
        return pd.get_decomposition(composition)
    except IndexError:
        raise ValueError('Error during PD building; most likely, cost data does not exist!')