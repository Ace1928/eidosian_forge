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
def read_hessian(self, noproj=False):
    """Read in the hessian matrix"""
    self.results['hessian matrix'] = {}
    self.results['hessian matrix']['array'] = []
    self.results['hessian matrix']['units'] = '?'
    self.results['hessian matrix']['projected'] = True
    self.results['hessian matrix']['mass weighted'] = True
    dg = read_data_group('nvibro')
    if len(dg) == 0:
        return
    nvibro = int(dg.split()[1])
    self.results['hessian matrix']['dimension'] = nvibro
    row = []
    key = 'hessian'
    if noproj:
        key = 'npr' + key
        self.results['hessian matrix']['projected'] = False
    lines = read_data_group(key).split('\n')
    for line in lines:
        if key in line:
            continue
        fields = line.split()
        row.extend(fields[2:len(fields)])
        if len(row) == nvibro:
            float_row = [float(element) for element in row]
            self.results['hessian matrix']['array'].append(float_row)
            row = []