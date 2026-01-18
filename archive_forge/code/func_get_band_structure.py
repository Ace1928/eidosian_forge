import os
import time
import subprocess
import re
import warnings
import numpy as np
from ase.geometry import cell_to_cellpar
from ase.calculators.calculator import (FileIOCalculator, Calculator, equal,
from ase.calculators.openmx.parameters import OpenMXParameters
from ase.calculators.openmx.default_settings import default_dictionary
from ase.calculators.openmx.reader import read_openmx, get_file_name
from ase.calculators.openmx.writer import write_openmx
def get_band_structure(self, atoms=None, calc=None):
    """
        This is band structure function. It is compatible to
        ase dft module """
    from ase.dft import band_structure
    if type(self['kpts']) is tuple:
        self['kpts'] = self.get_kpoints(band_kpath=self['band_kpath'])
        return band_structure.get_band_structure(self.atoms, self)