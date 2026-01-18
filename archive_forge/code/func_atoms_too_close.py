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
def atoms_too_close(atoms, bl, use_tags=False):
    """Checks if any atoms in a are too close, as defined by
    the distances in the bl dictionary.

    use_tags: whether to use the Atoms tags to disable distance
        checking within a set of atoms with the same tag.

    Note: if certain atoms are constrained and use_tags is True,
    this method may return unexpected results in case the
    contraints prevent same-tag atoms to be gathered together in
    the minimum-image-convention. In such cases, one should
    (1) release the relevant constraints,
    (2) apply the gather_atoms_by_tag function, and
    (3) re-apply the constraints, before using the
        atoms_too_close function.
    """
    a = atoms.copy()
    if use_tags:
        gather_atoms_by_tag(a)
    pbc = a.get_pbc()
    cell = a.get_cell()
    num = a.get_atomic_numbers()
    pos = a.get_positions()
    tags = a.get_tags()
    unique_types = sorted(list(set(num)))
    neighbours = []
    for i in range(3):
        if pbc[i]:
            neighbours.append([-1, 0, 1])
        else:
            neighbours.append([0])
    for nx, ny, nz in itertools.product(*neighbours):
        displacement = np.dot(cell.T, np.array([nx, ny, nz]).T)
        pos_new = pos + displacement
        distances = cdist(pos, pos_new)
        if nx == 0 and ny == 0 and (nz == 0):
            if use_tags and len(a) > 1:
                x = np.array([tags]).T
                distances += 100.0 * (cdist(x, x) == 0)
            else:
                distances += 100.0 * np.identity(len(a))
        iterator = itertools.combinations_with_replacement(unique_types, 2)
        for type1, type2 in iterator:
            x1 = np.where(num == type1)
            x2 = np.where(num == type2)
            if np.min(distances[x1].T[x2]) < bl[type1, type2]:
                return True
    return False