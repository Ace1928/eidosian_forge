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
def read_pld(self, norb, natms):
    """
        Read the siesta PLD file
        Return a namedtuple with the following arguments:
        'max_rcut', 'orb2ao', 'orb2uorb', 'orb2occ', 'atm2sp',
        'atm2shift', 'coord_sc', 'cell', 'nunit_cells'
        """
    from ase.calculators.siesta.import_functions import readPLD
    filename = self.getpath(ext='PLD')
    if isfile(filename):
        self.results['pld'] = readPLD(filename, norb, natms)
    else:
        self.results['pld'] = None