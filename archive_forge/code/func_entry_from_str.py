from and back to a string/file is not guaranteed to be reversible, i.e. a diff on the output
from __future__ import annotations
import datetime
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element, Lattice, PeriodicSite, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.core import ParseError
@classmethod
def entry_from_str(cls, string: str) -> ComputedStructureEntry:
    """Produce a pymatgen ComputedStructureEntry from contents of a res file."""
    return AirssProvider.from_str(string).entry