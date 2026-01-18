import math
from typing import Optional, Sequence
import numpy as np
from ase.atoms import Atoms
import ase.data
def convert_to_natural_basis(self):
    """Convert directions and miller indices to the natural basis."""
    self.directions = np.dot(self.directions, self.inverse_basis)
    if self.bravais_basis is not None:
        self.natural_bravais_basis = np.dot(self.bravais_basis, self.inverse_basis)
    for i in (0, 1, 2):
        self.directions[i] = reduceindex(self.directions[i])
    for i in (0, 1, 2):
        j, k = self.other[i]
        self.miller[i] = reduceindex(self.handedness * cross(self.directions[j], self.directions[k]))