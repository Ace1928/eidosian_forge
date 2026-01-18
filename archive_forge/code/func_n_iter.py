from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def n_iter(self) -> int | None:
    """The number of steps needed to converge the SCF cycle for the chunk."""
    val = self.parse_scalar('number_of_iterations')
    if val is not None:
        return int(val)
    return None