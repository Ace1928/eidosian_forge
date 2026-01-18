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
def ensure_eigenmode_orthogonality(self, order):
    mode = self.eigenmodes[order - 1].copy()
    for k in range(order - 1):
        mode = perpendicular_vector(mode, self.eigenmodes[k])
    self.eigenmodes[order - 1] = normalize(mode)