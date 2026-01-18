import re
import numpy as np
from ase.units import Bohr, Angstrom, Hartree, eV, Debye
def get_energy_unit(line):
    return {'[eV]': eV, '[H]': Hartree}[line.split()[1].rstrip(':')]