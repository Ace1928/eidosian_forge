from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
def prepare_jacobian(self, pos):
    v0, v1, v2 = self.gather_vectors(pos)
    derivs = get_dihedrals_derivatives(v0, v1, v2, cell=self.cell, pbc=self.pbc)
    self.finalize_jacobian(pos, len(v0), 4, derivs)