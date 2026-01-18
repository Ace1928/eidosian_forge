import os
import time
import math
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from ase.io import write, read
from ase.geometry.cell import cell_to_cellpar
from ase.data import covalent_radii
from ase.ga import get_neighbor_list
def atoms_too_close_two_sets(a, b, bl):
    """Checks if any atoms in a are too close to an atom in b,
    as defined by the bl dictionary."""
    pbc_a = a.get_pbc()
    pbc_b = b.get_pbc()
    cell_a = a.get_cell()
    cell_b = a.get_cell()
    assert np.allclose(pbc_a, pbc_b), (pbc_a, pbc_b)
    assert np.allclose(cell_a, cell_b), (cell_a, cell_b)
    pos_a = a.get_positions()
    pos_b = b.get_positions()
    num_a = a.get_atomic_numbers()
    num_b = b.get_atomic_numbers()
    unique_types = sorted(set(list(num_a) + list(num_b)))
    neighbours = []
    for i in range(3):
        neighbours.append([-1, 0, 1] if pbc_a[i] else [0])
    for nx, ny, nz in itertools.product(*neighbours):
        displacement = np.dot(cell_a.T, np.array([nx, ny, nz]).T)
        pos_b_disp = pos_b + displacement
        distances = cdist(pos_a, pos_b_disp)
        for type1 in unique_types:
            if type1 not in num_a:
                continue
            x1 = np.where(num_a == type1)
            for type2 in unique_types:
                if type2 not in num_b:
                    continue
                x2 = np.where(num_b == type2)
                if np.min(distances[x1].T[x2]) < bl[type1, type2]:
                    return True
    return False