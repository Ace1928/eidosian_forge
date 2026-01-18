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
def read_mos(self):
    """read the molecular orbital coefficients and orbital energies
        from files mos, alpha and beta"""
    self.results['molecular orbitals'] = []
    mos = self.results['molecular orbitals']
    keywords = ['scfmo', 'uhfmo_alpha', 'uhfmo_beta']
    spin = [None, 'alpha', 'beta']
    for index, keyword in enumerate(keywords):
        flen = None
        mo = {}
        orbitals_coefficients_line = []
        mo_string = read_data_group(keyword)
        if mo_string == '':
            continue
        mo_string += '\n$end'
        lines = mo_string.split('\n')
        for line in lines:
            if re.match('^\\s*#', line):
                continue
            if 'eigenvalue' in line:
                if len(orbitals_coefficients_line) != 0:
                    mo['eigenvector'] = orbitals_coefficients_line
                    mos.append(mo)
                    mo = {}
                    orbitals_coefficients_line = []
                regex = '^\\s*(\\d+)\\s+(\\S+)\\s+eigenvalue=([\\+\\-\\d\\.\\w]+)\\s'
                match = re.search(regex, line)
                mo['index'] = int(match.group(1))
                mo['irreducible representation'] = str(match.group(2))
                eig = float(re.sub('[dD]', 'E', match.group(3))) * Ha
                mo['eigenvalue'] = eig
                mo['spin'] = spin[index]
                mo['degeneracy'] = 1
                continue
            if keyword in line:
                regex = 'format\\(\\d+[a-zA-Z](\\d+)\\.\\d+\\)'
                match = re.search(regex, line)
                if match:
                    flen = int(match.group(1))
                if 'scfdump' in line or 'expanded' in line or 'scfconv' not in line:
                    self.converged = False
                continue
            if '$end' in line:
                if len(orbitals_coefficients_line) != 0:
                    mo['eigenvector'] = orbitals_coefficients_line
                    mos.append(mo)
                break
            sfields = [line[i:i + flen] for i in range(0, len(line), flen)]
            ffields = [float(f.replace('D', 'E').replace('d', 'E')) for f in sfields]
            orbitals_coefficients_line += ffields