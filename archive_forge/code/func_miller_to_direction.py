import math
from typing import Optional, Sequence
import numpy as np
from ase.atoms import Atoms
import ase.data
def miller_to_direction(self, miller):
    """Returns the direction corresponding to a given Miller index."""
    return np.dot(miller, self.millerbasis)