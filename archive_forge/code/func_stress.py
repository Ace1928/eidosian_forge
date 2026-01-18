from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def stress(self) -> Matrix3D | None:
    """The stress from the aims.out file and convert to kbar."""
    line_start = self.reverse_search_for(['Analytical stress tensor - Symmetrized', 'Numerical stress tensor'])
    if line_start == LINE_NOT_FOUND:
        return None
    stress = [[float(inp) for inp in line.split()[2:5]] for line in self.lines[line_start + 5:line_start + 8]]
    return np.array(stress) * EV_PER_A3_TO_KBAR