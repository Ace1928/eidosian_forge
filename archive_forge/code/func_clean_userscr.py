import os
import re
from subprocess import call, TimeoutExpired
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.utils import workdir
from ase.units import Hartree, Bohr, Debye
from ase.calculators.singlepoint import SinglePointCalculator
def clean_userscr(userscr, prefix):
    for fname in os.listdir(userscr):
        tokens = fname.split('.')
        if tokens[0] == prefix and tokens[-1] != 'bak':
            fold = os.path.join(userscr, fname)
            os.rename(fold, fold + '.bak')