from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def magmom(self) -> float | None:
    """The magnetic moment of the structure"""
    return self.parse_scalar('magnetic_moment')