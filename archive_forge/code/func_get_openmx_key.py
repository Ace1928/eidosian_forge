import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def get_openmx_key(key):
    """
    For the writing purpose, we need to know Original OpenMX keyword format.
    By comparing keys in the parameters.py, restore the original key
    """
    for openmx_key in keys:
        for openmx_keyword in openmx_key:
            if key == get_standard_key(openmx_keyword):
                return openmx_keyword