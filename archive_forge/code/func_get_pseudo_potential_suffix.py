import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def get_pseudo_potential_suffix(element=None, xc=None, year='13'):
    """
    For a given element, returns the string specifying pseudo potential suffix.
    For example,
        'Si'   ->   'Si_CA13'
    or
        'Si'   ->   'Si_CA19'
    depending on pseudo potential generation year
    """
    from ase.calculators.openmx import default_settings
    default_dictionary = default_settings.default_dictionary
    pseudo_potential_suffix = element
    vps = get_vps(xc)
    suffix = default_dictionary[element]['pseudo-potential suffix']
    pseudo_potential_suffix += '_' + vps + year + suffix
    return pseudo_potential_suffix