import numpy as np
from numpy import linalg
from ase import units 
def get_coulomb_potential_gradient(atoms, coulomb):
    i = coulomb.atomi
    j = coulomb.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    eij = rij / dij
    gr = -coulomb.chargeij / dij / dij * eij
    gx = np.dot(Bx.T, gr)
    coulomb.r = dij
    return (i, j, gx)