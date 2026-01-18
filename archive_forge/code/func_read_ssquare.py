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
def read_ssquare(self):
    """Read the expectation value of S^2 operator"""
    s2_string = read_data_group('ssquare from dscf')
    if s2_string == '':
        return
    string = s2_string.split('\n')[1]
    ssquare = float(re.search('^\\s*(\\d+\\.*\\d*)', string).group(1))
    self.results['ssquare from scf calculation'] = ssquare