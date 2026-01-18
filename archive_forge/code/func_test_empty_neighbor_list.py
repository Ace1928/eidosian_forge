import numpy.random as random
import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import (NeighborList, PrimitiveNeighborList,
from ase.build import bulk
def test_empty_neighbor_list():
    nl = PrimitiveNeighborList([])
    nl.update([True, True, True], np.eye(3) * 7.56, np.zeros((0, 3)))