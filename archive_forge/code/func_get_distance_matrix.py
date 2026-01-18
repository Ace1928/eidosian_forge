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
def get_distance_matrix(atoms, self_distance=1000):
    """NB: This function is way slower than atoms.get_all_distances()

    Returns a numpy matrix with the distances between the atoms
    in the supplied atoms object, with the indices of the matrix
    corresponding to the indices in the atoms object.

    The parameter self_distance will be put in the diagonal
    elements ([i][i])
    """
    dm = np.zeros([len(atoms), len(atoms)])
    for i in range(len(atoms)):
        dm[i][i] = self_distance
        for j in range(i + 1, len(atoms)):
            rij = atoms.get_distance(i, j)
            dm[i][j] = rij
            dm[j][i] = rij
    return dm