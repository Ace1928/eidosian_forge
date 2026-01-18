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
def read_hsx(self):
    """
        Read the siesta HSX file.
        return a namedtuple with the following arguments:
        'norbitals', 'norbitals_sc', 'nspin', 'nonzero',
        'is_gamma', 'sc_orb2uc_orb', 'row2nnzero', 'sparse_ind2column',
        'H_sparse', 'S_sparse', 'aB2RaB_sparse', 'total_elec_charge', 'temp'
        """
    from ase.calculators.siesta.import_functions import readHSX
    filename = self.getpath(ext='HSX')
    if isfile(filename):
        self.results['hsx'] = readHSX(filename)
    else:
        self.results['hsx'] = None