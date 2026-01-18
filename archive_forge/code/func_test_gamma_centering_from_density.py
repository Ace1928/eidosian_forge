import numpy as np
from ase import Atoms
from ase.calculators.calculator import kpts2sizeandoffsets as k2so
def test_gamma_centering_from_density():
    kd = 24 / (2 * np.pi)
    size, offsets = map(tuple, k2so(density=kd, gamma=True, atoms=Atoms(cell=(a, a, a), pbc=True)))
    assert (size, offsets) == ((4, 4, 4), (0.125, 0.125, 0.125))