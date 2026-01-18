from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def k_points(self) -> Sequence[Vector3D]:
    """All k-points listed in the calculation."""
    return self._header['k_points']