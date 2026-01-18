import numpy as np
from numpy import linalg
from ase import units 
def get_morse_potential_reduced_hessian(atoms, morse):
    i = morse.atomi
    j = morse.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    eij = rij / dij
    Pij = np.tensordot(eij, eij, axes=0)
    exp = np.exp(-morse.alpha * (dij - morse.r0))
    Hr = np.abs(2.0 * morse.D * morse.alpha ** 2 * exp * (2.0 * exp - 1.0)) * Pij
    Hx = np.dot(Mx.T, np.dot(Hr, Mx))
    morse.r = dij
    return (i, j, Hx)