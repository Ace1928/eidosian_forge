from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def stresses(self) -> np.array[Matrix3D] | None:
    """The stresses from the aims.out file and convert to kbar."""
    line_start = self.reverse_search_for(['Per atom stress (eV) used for heat flux calculation'])
    if line_start == LINE_NOT_FOUND:
        return None
    line_start += 3
    stresses = []
    for line in self.lines[line_start:line_start + self.n_atoms]:
        xx, yy, zz, xy, xz, yz = (float(d) for d in line.split()[2:8])
        stresses.append(Tensor.from_voigt([xx, yy, zz, yz, xz, xy]))
    return np.array(stresses) * EV_PER_A3_TO_KBAR