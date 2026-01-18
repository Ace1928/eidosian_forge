import numpy as np
from ase.neighborlist import NeighborList
from ase.data import covalent_radii
def get_bond_list(atoms, nl, rs):
    bonds = []
    for i in range(len(atoms)):
        p = atoms.positions[i]
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            q = atoms.positions[j] + np.dot(offset, atoms.get_cell())
            d = np.linalg.norm(p - q)
            k = d / (rs[i] + rs[j])
            bonds.append((k, i, j, tuple(offset)))
    return set(bonds)