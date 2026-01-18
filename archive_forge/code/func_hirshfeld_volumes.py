from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
@property
def hirshfeld_volumes(self) -> Sequence[float] | None:
    """The Hirshfeld atomic dipoles of the system"""
    if 'hirshfeld_volumes' not in self._cache:
        self._parse_hirshfeld()
    return self._cache['hirshfeld_volumes']