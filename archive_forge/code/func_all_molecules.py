from __future__ import annotations
import re
from io import StringIO
from typing import TYPE_CHECKING, cast
import pandas as pd
from monty.io import zopen
from pymatgen.core import Molecule, Structure
from pymatgen.core.structure import SiteCollection
@property
def all_molecules(self) -> list[Molecule]:
    """Returns all the frames of molecule associated with this XYZ."""
    return self._mols