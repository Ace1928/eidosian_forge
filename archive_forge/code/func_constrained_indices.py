from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
def constrained_indices(atoms, only_include=None):
    """Returns a list of indices for the atoms that are constrained
    by a constraint that is applied.  By setting only_include to a
    specific type of constraint you can make it only look for that
    given constraint.
    """
    indices = []
    for constraint in atoms.constraints:
        if only_include is not None:
            if not isinstance(constraint, only_include):
                continue
        indices.extend(np.array(constraint.get_indices()))
    return np.array(np.unique(indices))