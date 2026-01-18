import numpy as np
from ase import Atoms
from ase.calculators.calculator import kpts2sizeandoffsets as k2so
def test_antigamma_centering_from_default_111():
    size, offsets = map(tuple, k2so(gamma=False, atoms=Atoms(cell=(a, a, a), pbc=True)))
    assert (size, offsets) == ((1, 1, 1), (0.5, 0.5, 0.5))