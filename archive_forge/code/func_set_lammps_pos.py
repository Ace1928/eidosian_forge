import ctypes
import numpy as np
from numpy.linalg import norm
from ase.calculators.calculator import Calculator
from ase.data import (atomic_numbers as ase_atomic_numbers,
from ase.calculators.lammps import convert
from ase.geometry import wrap_positions
def set_lammps_pos(self, atoms):
    cell = convert(atoms.cell, 'distance', 'ASE', self.units)
    pos = convert(atoms.positions, 'distance', 'ASE', self.units)
    if self.coord_transform is not None:
        pos = np.dot(pos, self.coord_transform.T)
        cell = np.dot(cell, self.coord_transform.T)
    pos = wrap_positions(pos, cell, atoms.get_pbc())
    lmp_positions = list(pos.ravel())
    c_double_array = ctypes.c_double * len(lmp_positions)
    lmp_c_positions = c_double_array(*lmp_positions)
    self.lmp.scatter_atoms('x', 1, 3, lmp_c_positions)