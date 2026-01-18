from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.bandstructure import Kpoint
@property
def has_eigendisplacements(self) -> bool:
    """True if eigendisplacements are present."""
    return len(self.eigendisplacements) > 0