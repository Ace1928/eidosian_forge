import numpy as np
from numpy import linalg
from ase import units 
def get_bond_potential_reduced_hessian_test(atoms, bond):
    i, j, v = get_bond_potential_value(atoms, bond)
    i, j, gx = get_bond_potential_gradient(atoms, bond)
    Hx = np.tensordot(gx, gx, axes=0) / v / 2.0
    return (i, j, Hx)