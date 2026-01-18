import numpy.random as random
import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import (NeighborList, PrimitiveNeighborList,
from ase.build import bulk
def test_fcc():
    x = bulk('X', 'fcc', a=2 ** 0.5)
    nl = NeighborList([0.5], skin=0.01, bothways=True, self_interaction=False)
    nl.update(x)
    assert len(nl.get_neighbors(0)[0]) == 12
    nl = NeighborList([0.5] * 27, skin=0.01, bothways=True, self_interaction=False)
    nl.update(x * (3, 3, 3))
    for a in range(27):
        assert len(nl.get_neighbors(a)[0]) == 12
    assert not np.any(nl.get_neighbors(13)[1])
    c = 0.0058
    for NeighborListClass in [PrimitiveNeighborList, NewPrimitiveNeighborList]:
        nl = NeighborListClass([c, c], skin=0.0, sorted=True, self_interaction=False, use_scaled_positions=True)
        nl.update([True, True, True], np.eye(3) * 7.56, np.array([[0, 0, 0], [0, 0, 0.99875]]))
        n0, d0 = nl.get_neighbors(0)
        n1, d1 = nl.get_neighbors(1)
        assert (np.all(n0 == [0]) and np.all(d0 == [0, 0, 1])) != (np.all(n1 == [1]) and np.all(d1 == [0, 0, -1]))