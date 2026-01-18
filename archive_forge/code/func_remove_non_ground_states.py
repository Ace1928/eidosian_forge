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
def remove_non_ground_states(self):
    """Removes all non-ground state entries, i.e., only keep the lowest energy
        per atom entry at each composition.
        """
    self.entries = self.ground_states