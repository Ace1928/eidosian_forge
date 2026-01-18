import numpy as np
from numpy import linalg
from ase import units 
def get_morse_potential_eta(atoms, morse):
    i = morse.atomi
    j = morse.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    if dij > morse.r0:
        exp = np.exp(-morse.alpha * (dij - morse.r0))
        eta = 1.0 - (1.0 - exp) ** 2
    else:
        eta = 1.0
    return eta