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
def read_vibrational_reduced_masses(self):
    """Read vibrational reduced masses"""
    self.results['vibrational reduced masses'] = []
    dg = read_data_group('vibrational reduced masses')
    if len(dg) == 0:
        return
    lines = dg.split('\n')
    for line in lines:
        if '$vibrational' in line:
            continue
        if '$end' in line:
            break
        fields = [float(element) for element in line.split()]
        self.results['vibrational reduced masses'].extend(fields)