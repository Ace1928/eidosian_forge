import ase
from typing import Mapping, Sequence, Union
import numpy as np
from ase.utils.arraywrapper import arraylike
from ase.utils import pbc2pbc
def cartesian_positions(self, scaled_positions) -> np.ndarray:
    """Calculate Cartesian positions from scaled positions."""
    return scaled_positions @ self.complete()