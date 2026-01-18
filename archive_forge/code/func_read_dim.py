import os
import re
import tempfile
import warnings
import shutil
from os.path import join, isfile, islink
import numpy as np
from ase.units import Ry, eV, Bohr
from ase.data import atomic_numbers
from ase.io.siesta import read_siesta_xv
from ase.calculators.siesta.import_functions import read_rho
from ase.calculators.siesta.import_functions import \
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.siesta.parameters import PAOBasisBlock, Species
from ase.calculators.siesta.parameters import format_fdf
def read_dim(self):
    """
        Read the siesta DIM file
        Retrun a namedtuple with the following arguments:
        'natoms_sc', 'norbitals_sc', 'norbitals', 'nspin',
        'nnonzero', 'natoms_interacting'
        """
    from ase.calculators.siesta.import_functions import readDIM
    filename = self.getpath(ext='DIM')
    if isfile(filename):
        self.results['dim'] = readDIM(filename)
    else:
        self.results['dim'] = None