import string
import numpy as np
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
def energy_delta(atoms1, atoms2):
    E1 = atoms1.get_potential_energy()
    E2 = atoms2.get_potential_energy()
    return 'E1 = {:+.1E}, E2 = {:+.1E}, dE = {:+1.1E}'.format(E1, E2, E2 - E1)