from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def is_relaxation(self) -> bool:
    """Is the output for a relaxation?"""
    return self.reverse_search_for(['Geometry relaxation:']) != LINE_NOT_FOUND