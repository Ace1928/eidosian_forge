import sys
from typing import Dict, Any
import numpy as np
from ase.calculators.calculator import (get_calculator_class,
from ase.constraints import FixAtoms, UnitCellFilter
from ase.eos import EquationOfState
from ase.io import read, write, Trajectory
from ase.optimize import LBFGS
import ase.db as db
def calculate_once(self, atoms):
    args = self.args
    for p in args.properties or 'efsdMm':
        property, method = {'e': ('energy', 'get_potential_energy'), 'f': ('forces', 'get_forces'), 's': ('stress', 'get_stress'), 'd': ('dipole', 'get_dipole_moment'), 'M': ('magmom', 'get_magnetic_moment'), 'm': ('magmoms', 'get_magnetic_moments')}[p]
        try:
            getattr(atoms, method)()
        except PropertyNotImplementedError:
            pass