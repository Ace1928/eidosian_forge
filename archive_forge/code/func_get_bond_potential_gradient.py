import numpy as np
from numpy import linalg
from ase import units 
def get_bond_potential_gradient(atoms, bond):
    i = bond.atomi
    j = bond.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    eij = rij / dij
    gr = bond.k * (dij - bond.b0) * eij
    gx = np.dot(Bx.T, gr)
    bond.b = dij
    return (i, j, gx)