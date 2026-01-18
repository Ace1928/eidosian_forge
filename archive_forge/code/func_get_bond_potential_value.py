import numpy as np
from numpy import linalg
from ase import units 
def get_bond_potential_value(atoms, bond):
    i = bond.atomi
    j = bond.atomj
    rij = rel_pos_pbc(atoms, i, j)
    dij = linalg.norm(rij)
    v = 0.5 * bond.k * (dij - bond.b0) ** 2
    bond.b = dij
    return (i, j, v)