import os
import re
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.dmol import write_dmol_car, write_dmol_incoor
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
def read_spin_polarized(self):
    """Reads, from outmol file, if calculation is spin polarized."""
    lines = self._outmol_lines()
    for n, line in enumerate(lines):
        if line.rfind('Calculation is Spin_restricted') > -1:
            return False
        if line.rfind('Calculation is Spin_unrestricted') > -1:
            return True
    raise IOError('Could not read spin restriction from outmol')