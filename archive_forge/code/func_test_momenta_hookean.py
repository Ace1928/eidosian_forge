import pytest
import numpy as np
from ase.constraints import Hookean, FixAtoms
from ase.build import molecule
def test_momenta_hookean(atoms):
    atoms.set_constraint(Hookean(0, 1, rt=1.0, k=10.0))
    atoms.set_momenta(np.zeros(atoms.get_momenta().shape))
    actual = atoms.get_momenta()
    desired = np.zeros(atoms.get_momenta().shape)
    assert (actual == desired).all()