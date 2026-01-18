import numpy as np
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase import Atoms
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = np.abs(array - value).argmin()
    return idx