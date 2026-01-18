from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
def initialize_bond_lengths(self, atoms):
    bondlengths = np.zeros((len(self.triples), 2))
    for i in range(len(self.triples)):
        bondlengths[i, 0] = atoms.get_distance(self.n_ind[i], self.o_ind[i], mic=True)
        bondlengths[i, 1] = atoms.get_distance(self.o_ind[i], self.m_ind[i], mic=True)
    return bondlengths