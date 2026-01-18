import ase
from typing import Mapping, Sequence, Union
import numpy as np
from ase.utils.arraywrapper import arraylike
from ase.utils import pbc2pbc
@property
def handedness(self) -> int:
    """Sign of the determinant of the matrix of cell vectors.

        1 for right-handed cells, -1 for left, and 0 for cells that
        do not span three dimensions."""
    return int(np.sign(np.linalg.det(self)))