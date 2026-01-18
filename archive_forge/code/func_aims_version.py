from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MontyDecoder, MSONable
from pymatgen.io.aims.parsers import (
@property
def aims_version(self) -> str:
    """The version of FHI-aims used for the calculation."""
    return self._metadata['version_number']