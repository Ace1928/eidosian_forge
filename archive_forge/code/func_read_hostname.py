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
def read_hostname(self):
    """read the hostname of the computer on which the calc has run"""
    hostnames = read_output('hostname is\\s+(.+)')
    if len(set(hostnames)) > 1:
        warnings.warn('runs on different hosts detected')
        self.hostname = list(set(hostnames))
    else:
        self.hostname = hostnames[0]