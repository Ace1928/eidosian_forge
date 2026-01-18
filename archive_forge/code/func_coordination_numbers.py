import numpy as np
import pytest
from ase.cluster.decahedron import Decahedron
from ase.cluster.icosahedron import Icosahedron
from ase.cluster.octahedron import Octahedron
from ase.neighborlist import neighbor_list
def coordination_numbers(atoms):
    return np.bincount(neighbor_list('i', atoms, 0.8 * a0))