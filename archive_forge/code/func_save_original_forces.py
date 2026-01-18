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
def save_original_forces(self, force_calculation=False):
    """Store the forces (and energy) of the original state."""
    if self.calc is not None:
        if hasattr(self.calc, 'calculation_required') and (not self.calc.calculation_required(self.atoms, ['energy', 'forces'])) or force_calculation:
            calc = SinglePointCalculator(self.atoms0, energy=self.atoms.get_potential_energy(), forces=self.atoms.get_forces())
            self.atoms0.calc = calc