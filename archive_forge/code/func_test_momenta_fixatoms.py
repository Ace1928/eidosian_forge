import pytest
import numpy as np
from ase.constraints import Hookean, FixAtoms
from ase.build import molecule
def test_momenta_fixatoms(atoms):
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.set_momenta(np.ones(atoms.get_momenta().shape))
    desired = np.ones(atoms.get_momenta().shape)
    desired[0] = 0.0
    actual = atoms.get_momenta()
    assert (actual == desired).all()