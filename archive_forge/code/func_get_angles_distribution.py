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
def get_angles_distribution(atoms, ang_grid=9):
    """Method to get the distribution of bond angles
    in bins (default 9) with bonds defined from
    the get_neighbor_list().
    """
    conn = get_neighbor_list(atoms)
    if conn is None:
        conn = get_neighborlist(atoms)
    bins = [0] * ang_grid
    for atom in atoms:
        for i in conn[atom.index]:
            for j in conn[atom.index]:
                if j != i:
                    a = atoms.get_angle(i, atom.index, j)
                    for k in range(ang_grid):
                        if (k + 1) * 180.0 / ang_grid > a > k * 180.0 / ang_grid:
                            bins[k] += 1
    for i in range(ang_grid):
        bins[i] /= 2.0
    return bins