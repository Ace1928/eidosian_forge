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
def get_cg_direction(self, direction):
    """Apply the Conjugate Gradient algorithm to the step direction."""
    if self.cg_init:
        self.cg_init = False
        self.direction_old = direction.copy()
        self.cg_direction = direction.copy()
    old_norm = np.vdot(self.direction_old, self.direction_old)
    if old_norm != 0.0:
        betaPR = np.vdot(direction, direction - self.direction_old) / old_norm
    else:
        betaPR = 0.0
    if betaPR < 0.0:
        betaPR = 0.0
    self.cg_direction = direction + self.cg_direction * betaPR
    self.direction_old = direction.copy()
    return self.cg_direction.copy()