from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MontyDecoder, MSONable
from pymatgen.io.aims.parsers import (
@property
def structure_summary(self) -> dict[str, Any]:
    """The summary of the material/molecule that the calculations represent."""
    return self._structure_summary