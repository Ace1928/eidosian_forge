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
def read_runtime(self):
    """read the total runtime of calculations"""
    hits = read_output('total wall-time\\s+:\\s+(\\d+.\\d+)\\s+seconds')
    if len(hits) == 0:
        warnings.warn('no turbomole runtimes detected')
        self.runtime = None
    else:
        self.runtime = np.sum([float(a) for a in hits])