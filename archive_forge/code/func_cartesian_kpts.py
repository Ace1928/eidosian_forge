import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def cartesian_kpts(self) -> np.ndarray:
    """Get Cartesian kpoints from this bandpath."""
    return self._scale(self.kpts)