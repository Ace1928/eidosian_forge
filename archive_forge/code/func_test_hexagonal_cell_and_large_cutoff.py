import numpy.random as random
import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import (NeighborList, PrimitiveNeighborList,
from ase.build import bulk
def test_hexagonal_cell_and_large_cutoff():
    pbc_c = np.array([True, True, True])
    cutoff_a = np.array([8.0, 8.0])
    cell_cv = np.array([[0.0, 3.37316113, 3.37316113], [3.37316113, 0.0, 3.37316113], [3.37316113, 3.37316113, 0.0]])
    spos_ac = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    nl = PrimitiveNeighborList(cutoff_a, skin=0.0, sorted=True, use_scaled_positions=True)
    nl2 = NewPrimitiveNeighborList(cutoff_a, skin=0.0, sorted=True, use_scaled_positions=True)
    nl.update(pbc_c, cell_cv, spos_ac)
    nl2.update(pbc_c, cell_cv, spos_ac)
    a0, offsets0 = nl.get_neighbors(0)
    b0 = np.zeros_like(a0)
    d0 = np.dot(spos_ac[a0] + offsets0 - spos_ac[0], cell_cv)
    a1, offsets1 = nl.get_neighbors(1)
    d1 = np.dot(spos_ac[a1] + offsets1 - spos_ac[1], cell_cv)
    b1 = np.ones_like(a1)
    a = np.concatenate([a0, a1])
    b = np.concatenate([b0, b1])
    d = np.concatenate([d0, d1])
    _a = np.concatenate([a, b])
    _b = np.concatenate([b, a])
    a = _a
    b = _b
    d = np.concatenate([d, -d])
    a0, offsets0 = nl2.get_neighbors(0)
    d0 = np.dot(spos_ac[a0] + offsets0 - spos_ac[0], cell_cv)
    b0 = np.zeros_like(a0)
    a1, offsets1 = nl2.get_neighbors(1)
    d1 = np.dot(spos_ac[a1] + offsets1 - spos_ac[1], cell_cv)
    b1 = np.ones_like(a1)
    a2 = np.concatenate([a0, a1])
    b2 = np.concatenate([b0, b1])
    d2 = np.concatenate([d0, d1])
    _a2 = np.concatenate([a2, b2])
    _b2 = np.concatenate([b2, a2])
    a2 = _a2
    b2 = _b2
    d2 = np.concatenate([d2, -d2])
    i = np.argsort(d[:, 0] + d[:, 1] * 100.0 + d[:, 2] * 10000.0 + a * 1000000.0)
    i2 = np.argsort(d2[:, 0] + d2[:, 1] * 100.0 + d2[:, 2] * 10000.0 + a2 * 1000000.0)
    assert np.all(a[i] == a2[i2])
    assert np.all(b[i] == b2[i2])
    assert np.allclose(d[i], d2[i2])