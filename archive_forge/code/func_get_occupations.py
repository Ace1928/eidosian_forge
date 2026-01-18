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
def get_occupations(self, kpt=0, spin=0):
    return self.read_eigenvalues(kpt, spin, 'occupations')