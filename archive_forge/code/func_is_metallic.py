from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def is_metallic(self) -> bool:
    """Is the system is metallic."""
    line_start = self.reverse_search_for(['material is metallic within the approximate finite broadening function (occupation_type)'])
    return line_start != LINE_NOT_FOUND