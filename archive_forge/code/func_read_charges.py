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
def read_charges(self):
    """read partial charges on atoms from an ESP fit"""
    epsfit_defined = 'esp fit' in self.parameters and self.parameters['esp fit'] is not None
    if epsfit_defined or len(read_data_group('esp_fit')) > 0:
        filename = 'ASE.TM.' + self.calculate_energy + '.out'
        with open(filename, 'r') as infile:
            lines = infile.readlines()
        oklines = None
        for n, line in enumerate(lines):
            if 'atom  radius/au   charge' in line:
                oklines = lines[n + 1:n + len(self.atoms) + 1]
        if oklines is not None:
            qm_charges = [float(line.split()[3]) for line in oklines]
            self.charges = np.array(qm_charges)