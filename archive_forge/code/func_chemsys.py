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
def chemsys(self) -> set:
    """
        Returns:
            set representing the chemical system, e.g., {"Li", "Fe", "P", "O"}.
        """
    chemsys = set()
    for e in self.entries:
        chemsys.update([el.symbol for el in e.composition])
    return chemsys