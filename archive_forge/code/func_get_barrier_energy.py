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
def get_barrier_energy(self):
    """The energy difference between the current and original states"""
    try:
        original_energy = self.get_original_potential_energy()
        dimer_energy = self.get_potential_energy()
        return dimer_energy - original_energy
    except RuntimeError:
        w = 'The potential energy is not available, without further ' + 'calculations, most likely at the original state.'
        warnings.warn(w, UserWarning)
        return np.nan