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
def restore_original_positions(self):
    """Restore the MinModeAtoms object positions to the original state."""
    self.atoms.set_positions(self.get_original_positions())