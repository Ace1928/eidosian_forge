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
def is_ground_state(self, entry) -> bool:
    """Boolean indicating whether a given Entry is a ground state."""
    return entry in self.ground_states