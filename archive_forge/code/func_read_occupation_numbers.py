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
def read_occupation_numbers(self):
    """read occupation numbers with module 'eiger' """
    if 'molecular orbitals' not in self.results.keys():
        return
    mos = self.results['molecular orbitals']
    args = ['eiger', '--all', '--pview']
    output = execute(args, error_test=False, stdout_tofile=False)
    lines = output.split('\n')
    for line in lines:
        regex = '^\\s+(\\d+)\\.*\\s+(\\w*)\\s+(\\d+)\\s+(\\S+)\\s+(\\d*\\.*\\d*)\\s+([-+]?\\d+\\.\\d*)'
        match = re.search(regex, line)
        if match:
            orb_index = int(match.group(3))
            if match.group(2) == 'a':
                spin = 'alpha'
            elif match.group(2) == 'b':
                spin = 'beta'
            else:
                spin = None
            ar_index = next((index for index, molecular_orbital in enumerate(mos) if molecular_orbital['index'] == orb_index and molecular_orbital['spin'] == spin))
            mos[ar_index]['index by energy'] = int(match.group(1))
            irrep = str(match.group(4))
            mos[ar_index]['irreducible representation'] = irrep
            if match.group(5) != '':
                mos[ar_index]['occupancy'] = float(match.group(5))
            else:
                mos[ar_index]['occupancy'] = float(0)