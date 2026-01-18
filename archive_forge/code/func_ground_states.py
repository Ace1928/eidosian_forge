from __future__ import annotations
import collections
import csv
import datetime
import itertools
import json
import logging
import multiprocessing as mp
import re
from typing import TYPE_CHECKING, Literal
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.analysis.structure_matcher import SpeciesComparator, StructureMatcher
from pymatgen.core import Composition, Element
@property
def ground_states(self) -> set:
    """A set containing only the entries that are ground states, i.e., the lowest energy
        per atom entry at each composition.
        """
    entries = sorted(self.entries, key=lambda e: e.reduced_formula)
    return {min(g, key=lambda e: e.energy_per_atom) for _, g in itertools.groupby(entries, key=lambda e: e.reduced_formula)}