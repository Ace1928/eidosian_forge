import os
import re
import warnings
from subprocess import Popen, PIPE
from math import log10, floor
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError, ReadError
def read_dipole_moment(self):
    """Read the dipole moment"""
    dip_string = read_data_group('dipole')
    if dip_string == '':
        return
    lines = dip_string.split('\n')
    for line in lines:
        regex = '^\\s+x\\s+([-+]?\\d+\\.\\d*)\\s+y\\s+([-+]?\\d+\\.\\d*)\\s+z\\s+([-+]?\\d+\\.\\d*)\\s+a\\.u\\.'
        match = re.search(regex, line)
        if match:
            dip_vec = [float(match.group(c)) for c in range(1, 4)]
        regex = '^\\s+\\| dipole \\| =\\s+(\\d+\\.*\\d*)\\s+debye'
        match = re.search(regex, line)
        if match:
            dip_abs_val = float(match.group(1))
    self.results['electric dipole moment'] = {}
    self.results['electric dipole moment']['vector'] = {'array': dip_vec, 'units': 'a.u.'}
    self.results['electric dipole moment']['absolute value'] = {'value': dip_abs_val, 'units': 'Debye'}
    self.dipole = np.array(dip_vec) * Bohr