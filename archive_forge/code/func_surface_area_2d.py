import numpy as np
from itertools import combinations_with_replacement
from math import erf
from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc
def surface_area_2d(r, pos):
    p0 = pos[non_pbc_dirs[0]]
    area = np.minimum(pmax - p0, r) + np.minimum(p0 - pmin, r)
    area *= 2 * np.pi * r
    return area