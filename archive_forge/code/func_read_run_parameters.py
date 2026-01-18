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
def read_run_parameters(self):
    """read parameters set by define and not in self.parameters"""
    if 'calculation parameters' not in self.results.keys():
        self.results['calculation parameters'] = {}
    parameters = self.results['calculation parameters']
    dg = read_data_group('symmetry')
    parameters['point group'] = str(dg.split()[1])
    parameters['uhf'] = '$uhf' in read_data_group('uhf')
    gt = read_data_group('pople')
    if gt == '':
        parameters['gaussian type'] = 'spherical harmonic'
    else:
        gt = gt.split()[1]
        if gt == 'AO':
            parameters['gaussian type'] = 'spherical harmonic'
        elif gt == 'CAO':
            parameters['gaussian type'] = 'cartesian'
        else:
            parameters['gaussian type'] = None
    nvibro = read_data_group('nvibro')
    if nvibro:
        parameters['nuclear degrees of freedom'] = int(nvibro.split()[1])