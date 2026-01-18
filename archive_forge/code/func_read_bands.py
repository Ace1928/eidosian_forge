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
def read_bands(self):
    bandpath = self['bandpath']
    if bandpath is None:
        return
    if len(bandpath.kpts) < 1:
        return
    fname = self.getpath(ext='bands')
    with open(fname) as fd:
        kpts, energies, efermi = read_bands_file(fd)
    bs = resolve_band_structure(bandpath, kpts, energies, efermi)
    self.results['bandstructure'] = bs