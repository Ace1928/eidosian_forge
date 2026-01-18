from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MontyDecoder, MSONable
from pymatgen.io.aims.parsers import (
@property
def final_energy(self) -> float:
    """The total energy for the final structure in the calculation."""
    return self.get_results_for_image(-1).properties['energy']