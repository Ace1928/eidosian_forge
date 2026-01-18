import os
import warnings
import time
from typing import Optional
import re
import numpy as np
from ase.units import Hartree
from ase.io.aims import write_aims, read_aims
from ase.data import atomic_numbers
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, \
def read_magnetic_moment(self):
    magmom = None
    if not self.get_spin_polarized():
        magmom = 0.0
    else:
        for line in open(self.out, 'r').readlines():
            if line.find('N_up - N_down') != -1:
                magmom = float(line.split(':')[-1].strip())
    return magmom