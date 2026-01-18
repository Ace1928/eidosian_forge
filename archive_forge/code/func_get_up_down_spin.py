import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def get_up_down_spin(magmom, element, xc, data_path, year):
    magmom = np.linalg.norm(magmom)
    suffix = get_pseudo_potential_suffix(element, xc, year)
    filename = os.path.join(data_path, 'VPS/' + suffix + '.vps')
    valence_electron = float(read_electron_valency(filename))
    return [valence_electron / 2 + magmom / 2, valence_electron / 2 - magmom / 2]