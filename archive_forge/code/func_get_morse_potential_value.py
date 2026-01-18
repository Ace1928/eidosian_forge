import numpy as np
from numpy import linalg
from ase import units 
def get_morse_potential_value(atoms, morse):
    i = morse.atomi
    j = morse.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    exp = np.exp(-morse.alpha * (dij - morse.r0))
    v = morse.D * (1.0 - exp) ** 2
    morse.r = dij
    return (i, j, v)