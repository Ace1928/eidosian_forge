import numpy as np
from numpy import linalg
from ase import units 
def rel_pos_pbc(atoms, i, j):
    """
    Return difference between two atomic positions, 
    correcting for jumps across PBC
    """
    d = atoms.get_positions()[i, :] - atoms.get_positions()[j, :]
    g = linalg.inv(atoms.get_cell().T)
    f = np.floor(np.dot(g, d.T) + 0.5)
    d -= np.dot(atoms.get_cell().T, f).T
    return d