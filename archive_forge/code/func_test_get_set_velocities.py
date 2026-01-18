import pytest
import numpy as np
from ase.constraints import Hookean, FixAtoms
from ase.build import molecule
def test_get_set_velocities(atoms):
    shape = (len(atoms), 3)
    assert np.array_equal(atoms.get_velocities(), np.zeros(shape))
    rng = np.random.RandomState(17)
    v0 = rng.rand(*shape)
    atoms.set_velocities(v0)
    assert atoms.get_velocities() == pytest.approx(v0)