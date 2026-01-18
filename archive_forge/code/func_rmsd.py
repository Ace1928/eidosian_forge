import string
import numpy as np
from ase.io import string2index
from ase.io.formats import parse_filename
from ase.data import chemical_symbols
def rmsd(atoms1, atoms2):
    dpositions = atoms2.positions - atoms1.positions
    return 'RMSD={:+.1E}'.format(np.sqrt((np.linalg.norm(dpositions, axis=1) ** 2).mean()))