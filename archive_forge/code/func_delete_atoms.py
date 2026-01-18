from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
def delete_atoms(self, indices, natoms):
    """Removes atom number ind from the index array, if present.

        Required for removing atoms with existing FixAtoms constraints.
        """
    i = np.zeros(natoms, int) - 1
    new = np.delete(np.arange(natoms), indices)
    i[new] = np.arange(len(new))
    index = i[self.index]
    self.index = index[index >= 0]
    if len(self.index) == 0:
        return None
    return self