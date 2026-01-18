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
def get_curvature(self, order='max'):
    """Return the eigenvalue estimate."""
    if order == 'max':
        return max(self.curvatures)
    else:
        return self.curvatures[order - 1]