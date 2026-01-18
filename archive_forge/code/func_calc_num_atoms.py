import math
from typing import Optional, Sequence
import numpy as np
from ase.atoms import Atoms
import ase.data
def calc_num_atoms(self):
    v = int(round(abs(np.linalg.det(self.directions))))
    if self.bravais_basis is None:
        return v
    else:
        return v * len(self.bravais_basis)