import sys
import time
import warnings
from math import cos, sin, atan, tan, degrees, pi, sqrt
from typing import Dict, Any
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.parallel import world
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import IOContext
def get_rotational_force(self):
    """Calculate the rotational force that acts on the dimer."""
    rot_force = perpendicular_vector(self.forces1 - self.forces2, self.eigenmode) / (2.0 * self.dR)
    if self.basis is not None:
        if len(self.basis) == len(self.atoms) and len(self.basis[0]) == 3 and isinstance(self.basis[0][0], float):
            rot_force = perpendicular_vector(rot_force, self.basis)
        else:
            for base in self.basis:
                rot_force = perpendicular_vector(rot_force, base)
    return rot_force