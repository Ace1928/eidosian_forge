import ctypes
import numpy as np
from numpy.linalg import norm
from ase.calculators.calculator import Calculator
from ase.data import (atomic_numbers as ase_atomic_numbers,
from ase.calculators.lammps import convert
from ase.geometry import wrap_positions
def lammpsbc(self, atoms):
    """Determine LAMMPS boundary types based on ASE pbc settings. For
        non-periodic dimensions, if the cell length is finite then
        fixed BCs ('f') are used; if the cell length is approximately
        zero, shrink-wrapped BCs ('s') are used."""
    retval = ''
    pbc = atoms.get_pbc()
    if np.all(pbc):
        retval = 'p p p'
    else:
        cell = atoms.get_cell()
        for i in range(0, 3):
            if pbc[i]:
                retval += 'p '
            elif np.linalg.norm(cell[i]) < np.finfo(cell[i][0]).tiny:
                retval += 's '
            else:
                retval += 'f '
    return retval.strip()