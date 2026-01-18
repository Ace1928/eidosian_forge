import numpy as np
from ase import Atoms
from ase.calculators.calculator import kpts2sizeandoffsets as k2so
def test_density_with_irregular_shape():
    cell = [[2, 1, 0], [1, 2, 2], [-1, 0, 2]]
    kd = 3
    size, offsets = map(tuple, k2so(density=kd, atoms=Atoms(cell=cell, pbc=True)))
    assert (size, offsets) == ((29, 22, 26), (0, 0, 0))
    size, offsets = map(tuple, k2so(density=kd, even=True, atoms=Atoms(cell=cell, pbc=True)))
    assert (size, offsets) == ((30, 22, 26), (0, 0, 0))
    size, offsets = map(tuple, k2so(density=kd, even=True, gamma=True, atoms=Atoms(cell=cell, pbc=True)))
    assert (size, offsets) == ((30, 22, 26), (1 / 60, 1 / 44, 1 / 52))
    size, offsets = map(tuple, k2so(density=kd, even=False, atoms=Atoms(cell=cell, pbc=True)))
    assert (size, offsets) == ((29, 23, 27), (0, 0, 0))
    size, offsets = map(tuple, k2so(size=(3, 4, 5), even=True))
    assert (size, offsets) == ((4, 4, 6), (0, 0, 0))
    size, offsets = map(tuple, k2so(size=(3, 4, 5), even=False))
    assert (size, offsets) == ((3, 5, 5), (0, 0, 0))
    size, offsets = map(tuple, k2so(size=(5, 5, 1), gamma=False, atoms=Atoms(cell=(a, a, a), pbc=[True, True, False])))
    assert (size, offsets) == ((5, 5, 1), (0.1, 0.1, 0.0))