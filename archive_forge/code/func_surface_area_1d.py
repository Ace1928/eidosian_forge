import numpy as np
from itertools import combinations_with_replacement
from math import erf
from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc
def surface_area_1d(r, pos):
    q0 = pos[non_pbc_dirs[1]]
    phi1 = np.lib.scimath.arccos((qmax - q0) / r).real
    phi2 = np.pi - np.lib.scimath.arccos((qmin - q0) / r).real
    factor = 1 - (phi1 + phi2) / np.pi
    return surface_area_2d(r, pos) * factor