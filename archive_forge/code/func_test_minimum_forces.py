import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
def test_minimum_forces():
    for atoms in systems_minimum():
        np.testing.assert_allclose(atoms.get_forces(), 0, atol=1e-14)