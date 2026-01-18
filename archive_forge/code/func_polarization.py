from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def polarization(self) -> Vector3D | None:
    """The polarization vector from the aims.out file."""
    line_start = self.reverse_search_for(['| Cartesian Polarization'])
    if line_start == LINE_NOT_FOUND:
        return None
    line = self.lines[line_start]
    return np.array([float(s) for s in line.split()[-3:]])