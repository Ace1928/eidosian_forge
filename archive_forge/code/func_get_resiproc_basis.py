from typing import List, Optional
import numpy as np
from ase.data import atomic_numbers as ref_atomic_numbers
from ase.spacegroup import Spacegroup
from ase.cluster.base import ClusterBase
from ase.cluster.cluster import Cluster
def get_resiproc_basis(self, basis):
    """Returns the resiprocal basis to a given lattice (crystal) basis"""
    k = 1 / np.dot(basis[0], cross(basis[1], basis[2]))
    return k * np.array([cross(basis[1], basis[2]), cross(basis[2], basis[0]), cross(basis[0], basis[1])])