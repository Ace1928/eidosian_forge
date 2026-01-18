import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
def test_bulk_energies():
    for atoms in systems_bulk():
        assert np.allclose(atoms.get_potential_energy(), atoms.get_potential_energies().sum())
        assert atoms.get_potential_energies().std() == pytest.approx(0.0)