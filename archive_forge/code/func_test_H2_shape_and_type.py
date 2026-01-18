import numpy.random as random
import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import (NeighborList, PrimitiveNeighborList,
from ase.build import bulk
def test_H2_shape_and_type():
    h2 = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1)])
    nl = NeighborList([0.1, 0.1], skin=0.1, bothways=True, self_interaction=False)
    assert nl.update(h2)
    assert nl.get_neighbors(0)[1].shape == (0, 3)
    assert nl.get_neighbors(0)[1].dtype == int