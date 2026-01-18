from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def k_point_weights(self) -> Sequence[float]:
    """The k-point weights for the calculation."""
    return self._header['k_point_weights']