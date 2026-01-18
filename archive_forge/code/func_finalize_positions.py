from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
def finalize_positions(self, newpos):
    jacobian = self.jacobian / self.masses
    lamda = -self.sigma / np.dot(jacobian, self.jacobian)
    dnewpos = lamda * jacobian
    newpos += dnewpos.reshape(newpos.shape)